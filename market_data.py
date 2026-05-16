"""
Market Data Service
- Zerodha Kite WebSocket for live tick streaming (primary)
- Angel One SmartAPI as fallback
- Yahoo Finance for historical OHLCV (free, no auth)
- Redis pub/sub for broadcasting to WebSocket clients
- TimescaleDB for tick persistence
"""

import asyncio
import json
import httpx
import redis.asyncio as aioredis
from datetime import datetime, date, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ── Yahoo Finance OHLCV (free, no API key) ────────────────────────────────────

class YahooFinanceFetcher:
    """
    Fetch historical OHLCV for NSE stocks.
    Append .NS suffix for NSE: RELIANCE.NS, NIFTY50 → ^NSEI
    """
    BASE = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

    NIFTY_MAP = {
        "NIFTY":     "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "SENSEX":    "^BSESN",
    }

    @classmethod
    def _ticker(cls, symbol: str) -> str:
        s = symbol.upper()
        return cls.NIFTY_MAP.get(s, f"{s}.NS")

    @classmethod
    async def fetch_ohlcv(
        cls,
        symbol: str,
        period: str = "1y",        # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y
        interval: str = "1d",      # 1m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo
    ) -> list[dict]:
        ticker = cls._ticker(symbol)
        url    = cls.BASE.format(ticker=ticker)
        params = {"period1": 0, "period2": 9999999999,
                  "range": period, "interval": interval,
                  "includePrePost": "false"}
        headers = {"User-Agent": "Mozilla/5.0"}

        async with httpx.AsyncClient(headers=headers, timeout=15) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            raw = resp.json()

        result = raw["chart"]["result"][0]
        ts     = result["timestamp"]
        ohlcv  = result["indicators"]["quote"][0]
        adjclose = result["indicators"].get("adjclose", [{}])[0].get("adjclose", ohlcv["close"])

        rows = []
        for i, t in enumerate(ts):
            if ohlcv["close"][i] is None:
                continue
            rows.append({
                "time":   datetime.utcfromtimestamp(t).isoformat(),
                "symbol": symbol.upper(),
                "open":   round(ohlcv["open"][i] or 0, 2),
                "high":   round(ohlcv["high"][i] or 0, 2),
                "low":    round(ohlcv["low"][i] or 0, 2),
                "close":  round(ohlcv["close"][i] or 0, 2),
                "volume": int(ohlcv["volume"][i] or 0),
            })
        return rows

    @classmethod
    async def seed_database(cls, symbols: list[str], db_session, period: str = "2y"):
        """Bulk seed historical data into TimescaleDB on first run."""
        from sqlalchemy import text
        for symbol in symbols:
            try:
                rows = await cls.fetch_ohlcv(symbol, period=period)
                if not rows:
                    continue
                # Bulk insert via raw SQL for performance
                values = ", ".join(
                    f"('{r['time']}', '{r['symbol']}', {r['open']}, {r['high']}, "
                    f"{r['low']}, {r['close']}, {r['volume']})"
                    for r in rows
                )
                await db_session.execute(text(f"""
                    INSERT INTO tick_data (time, symbol, open, high, low, close, volume)
                    VALUES {values}
                    ON CONFLICT (time, symbol) DO NOTHING
                """))
                await db_session.commit()
                logger.info(f"Seeded {len(rows)} bars for {symbol}")
            except Exception as e:
                logger.error(f"Failed to seed {symbol}: {e}")


# ── Zerodha Kite WebSocket (live ticks) ──────────────────────────────────────

class KiteTickerService:
    """
    Wraps the kiteconnect KiteTicker for live tick streaming.
    Requires: pip install kiteconnect
    Publishes ticks to Redis channels for WebSocket clients.
    """

    def __init__(self, api_key: str, access_token: str):
        self.api_key      = api_key
        self.access_token = access_token
        self._redis: Optional[aioredis.Redis] = None

    async def _get_redis(self):
        if self._redis is None:
            self._redis = aioredis.from_url("redis://localhost:6379")
        return self._redis

    async def _publish_tick(self, tick: dict):
        r = await self._get_redis()
        symbol = tick.get("tradingsymbol", "")
        ltp    = tick.get("last_price", 0)
        payload = json.dumps({
            "symbol": symbol,
            "ltp":    ltp,
            "open":   tick.get("ohlc", {}).get("open", ltp),
            "high":   tick.get("ohlc", {}).get("high", ltp),
            "low":    tick.get("ohlc", {}).get("low", ltp),
            "close":  tick.get("ohlc", {}).get("close", ltp),
            "volume": tick.get("volume", 0),
            "oi":     tick.get("oi", 0),
            "time":   datetime.utcnow().isoformat(),
        })
        await r.set(f"ltp:{symbol}", ltp, ex=86400)       # cache LTP
        await r.publish(f"tick:{symbol}", payload)         # broadcast

    def start(self, instrument_tokens: list[int]):
        """
        Call this from a thread; KiteTicker is synchronous internally.
        Wraps callbacks to push into asyncio event loop.
        """
        try:
            from kiteconnect import KiteTicker
        except ImportError:
            logger.warning("kiteconnect not installed. Install: pip install kiteconnect")
            return

        loop = asyncio.get_event_loop()
        kt   = KiteTicker(self.api_key, self.access_token)

        def on_ticks(ws, ticks):
            for tick in ticks:
                asyncio.run_coroutine_threadsafe(self._publish_tick(tick), loop)

        def on_connect(ws, response):
            ws.subscribe(instrument_tokens)
            ws.set_mode(ws.MODE_FULL, instrument_tokens)

        def on_error(ws, code, reason):
            logger.error(f"KiteTicker error {code}: {reason}")

        def on_close(ws, code, reason):
            logger.info(f"KiteTicker closed: {code} {reason}")

        kt.on_ticks   = on_ticks
        kt.on_connect = on_connect
        kt.on_error   = on_error
        kt.on_close   = on_close
        kt.connect(threaded=True)


# ── Angel One SmartAPI (fallback / alternative) ───────────────────────────────

class AngelOneService:
    """
    REST API fallback for LTP when WebSocket is unavailable.
    Requires: pip install smartapi-python
    """
    BASE = "https://apiconnect.angelbroking.com"

    def __init__(self, api_key: str, client_id: str, password: str, totp_secret: str):
        self.api_key     = api_key
        self.client_id   = client_id
        self.password    = password
        self.totp_secret = totp_secret
        self._token: Optional[str] = None

    async def _login(self):
        import pyotp
        totp = pyotp.TOTP(self.totp_secret).now()
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.BASE}/rest/auth/angelbroking/user/v1/loginByPassword",
                json={
                    "clientcode": self.client_id,
                    "password":   self.password,
                    "totp":       totp,
                },
                headers={
                    "X-ClientID":    self.client_id,
                    "X-PrivateKey":  self.api_key,
                    "X-SourceID":    "WEB",
                    "X-UserType":    "USER",
                    "Content-Type":  "application/json",
                    "Accept":        "application/json",
                },
            )
            data = resp.json()
            self._token = data["data"]["jwtToken"]

    async def get_ltp(self, exchange: str, symbol: str, token: str) -> float:
        if not self._token:
            await self._login()
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.BASE}/rest/secure/angelbroking/market/v1/quote/",
                json={"mode": "LTP", "exchangeTokens": {exchange: [token]}},
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "X-PrivateKey":  self.api_key,
                    "Content-Type":  "application/json",
                    "Accept":        "application/json",
                },
            )
            data = resp.json()
            return float(data["data"]["fetched"][0]["ltp"])


# ── Orchestrator ──────────────────────────────────────────────────────────────

class MarketDataService:
    """
    Called from FastAPI lifespan to start background data tasks.
    """
    NIFTY50_SYMBOLS = [
        "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK",
        "HINDUNILVR","SBIN","BAJFINANCE","BHARTIARTL","KOTAKBANK",
        "LT","AXISBANK","ASIANPAINT","MARUTI","ULTRACEMCO",
        "TITAN","SUNPHARMA","WIPRO","ONGC","NTPC",
        "POWERGRID","NESTLEIND","TECHM","HCLTECH","JSWSTEEL",
        "TATAMOTORS","ADANIENT","GRASIM","INDUSINDBK","COALINDIA",
        "TATASTEEL","DIVISLAB","BPCL","CIPLA","DRREDDY",
        "EICHERMOT","APOLLOHOSP","BAJAJFINSV","HEROMOTOCO","HINDALCO",
        "M&M","BAJAJ-AUTO","SBILIFE","HDFCLIFE","UPL",
        "TATACONSUM","SHREECEM","BRITANNIA","ADANIPORTS","ITC",
    ]
    INDEX_SYMBOLS = ["NIFTY", "BANKNIFTY", "SENSEX"]

    @classmethod
    async def stream_nse_ticks(cls):
        """
        Polls Yahoo Finance every 15s during market hours as free-tier tick simulation.
        Replace with KiteTickerService for production real-time data.
        """
        import pytz
        IST = pytz.timezone("Asia/Kolkata")
        r   = aioredis.from_url("redis://localhost:6379")

        while True:
            try:
                now = datetime.now(IST)
                market_open  = now.replace(hour=9,  minute=15, second=0)
                market_close = now.replace(hour=15, minute=30, second=0)
                is_market_hours = market_open <= now <= market_close and now.weekday() < 5

                if is_market_hours:
                    symbols_to_poll = cls.NIFTY50_SYMBOLS[:10] + cls.INDEX_SYMBOLS
                    for sym in symbols_to_poll:
                        try:
                            rows = await YahooFinanceFetcher.fetch_ohlcv(sym, period="1d", interval="1m")
                            if rows:
                                latest = rows[-1]
                                payload = json.dumps({**latest, "source": "yahoo_1min"})
                                await r.set(f"ltp:{sym}", latest["close"], ex=300)
                                await r.publish(f"tick:{sym}", payload)
                        except Exception as e:
                            logger.debug(f"Tick poll failed for {sym}: {e}")
                    await asyncio.sleep(15)
                else:
                    await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"stream_nse_ticks error: {e}")
                await asyncio.sleep(30)

    @classmethod
    async def ingest_option_chain(cls):
        """Refresh option chain data every 3 minutes during market hours."""
        import pytz
        IST = pytz.timezone("Asia/Kolkata")
        r   = aioredis.from_url("redis://localhost:6379")

        while True:
            try:
                now = datetime.now(IST)
                market_open  = now.replace(hour=9,  minute=15, second=0)
                market_close = now.replace(hour=15, minute=30, second=0)
                is_market_hours = market_open <= now <= market_close and now.weekday() < 5

                if is_market_hours:
                    from services.options_engine import OptionChainFetcher
                    for idx in ["NIFTY", "BANKNIFTY"]:
                        try:
                            chain = await OptionChainFetcher.fetch(idx)
                            await r.set(
                                f"option_chain:{idx}",
                                json.dumps(chain),
                                ex=300,
                            )
                            logger.info(f"Option chain refreshed for {idx}")
                        except Exception as e:
                            logger.error(f"Option chain fetch failed for {idx}: {e}")
                    await asyncio.sleep(180)
                else:
                    await asyncio.sleep(120)
            except Exception as e:
                logger.error(f"ingest_option_chain error: {e}")
                await asyncio.sleep(60)
