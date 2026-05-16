"""
Technical Indicators Service
- RSI, MACD, Bollinger Bands, ATR, EMA, VWAP, SuperTrend
- Computed from TimescaleDB tick data
- Used by backtester and signal generator
"""

import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

try:
    import pandas_ta as ta
    PANDAS_TA = True
except ImportError:
    PANDAS_TA = False


class TechnicalIndicators:

    @staticmethod
    def compute(df: pd.DataFrame) -> pd.DataFrame:
        """
        df must have columns: open, high, low, close, volume
        Returns df with all indicator columns appended.
        """
        if df.empty:
            return df

        if PANDAS_TA:
            # RSI
            df["rsi"] = ta.rsi(df["close"], length=14)
            # MACD
            macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
            df["macd"]        = macd["MACD_12_26_9"]
            df["macd_signal"] = macd["MACDs_12_26_9"]
            df["macd_hist"]   = macd["MACDh_12_26_9"]
            # Bollinger Bands
            bb = ta.bbands(df["close"], length=20, std=2)
            df["bb_upper"] = bb["BBU_20_2.0"]
            df["bb_mid"]   = bb["BBM_20_2.0"]
            df["bb_lower"] = bb["BBL_20_2.0"]
            # ATR
            df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
            # EMAs
            df["ema_9"]  = ta.ema(df["close"], length=9)
            df["ema_20"] = ta.ema(df["close"], length=20)
            df["ema_50"] = ta.ema(df["close"], length=50)
            df["ema_200"]= ta.ema(df["close"], length=200)
            # VWAP (requires volume)
            df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
            # SuperTrend
            st = ta.supertrend(df["high"], df["low"], df["close"], length=10, multiplier=3.0)
            df["supertrend"]       = st["SUPERT_10_3.0"]
            df["supertrend_dir"]   = st["SUPERTd_10_3.0"]   # 1 = uptrend, -1 = downtrend
            # Stochastic
            stoch = ta.stoch(df["high"], df["low"], df["close"])
            df["stoch_k"] = stoch["STOCHk_14_3_3"]
            df["stoch_d"] = stoch["STOCHd_14_3_3"]
            # OBV
            df["obv"] = ta.obv(df["close"], df["volume"])

        else:
            # Manual fallback implementations
            df = TechnicalIndicators._manual_indicators(df)

        return df

    @staticmethod
    def _manual_indicators(df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high  = df["high"]
        low   = df["low"]
        vol   = df["volume"]

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["macd"]        = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"]   = df["macd"] - df["macd_signal"]

        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_mid"]   = sma20
        df["bb_lower"] = sma20 - 2 * std20

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

        # EMAs
        for span in [9, 20, 50, 200]:
            df[f"ema_{span}"] = close.ewm(span=span, adjust=False).mean()

        # VWAP (session-level; approximate daily)
        tp = (high + low + close) / 3
        df["vwap"] = (tp * vol).cumsum() / vol.cumsum()

        # OBV
        direction = np.sign(close.diff()).fillna(0)
        df["obv"] = (direction * vol).cumsum()

        return df

    @staticmethod
    def latest_snapshot(df: pd.DataFrame) -> dict:
        """Return the most recent bar's indicators as a flat dict for the signal agent."""
        if df.empty:
            return {}
        row = df.iloc[-1]
        cols = [
            "rsi", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_mid", "bb_lower",
            "atr", "ema_9", "ema_20", "ema_50", "ema_200",
            "vwap", "supertrend_dir", "stoch_k", "stoch_d", "obv",
        ]
        return {c: round(float(row[c]), 4) for c in cols if c in row.index and pd.notna(row[c])}


async def fetch_ohlcv(
    symbol: str,
    db: AsyncSession,
    days: int = 365,
    interval: str = "1D",
) -> pd.DataFrame:
    """
    Pull OHLCV from TimescaleDB tick_data table.
    interval: "1D" | "1H" | "15min"
    """
    since = datetime.utcnow() - timedelta(days=days)
    interval_map = {"1D": "1 day", "1H": "1 hour", "15min": "15 minutes"}
    bucket = interval_map.get(interval, "1 day")

    query = text(f"""
        SELECT
            time_bucket('{bucket}', time) AS ts,
            first(open,  time) AS open,
            max(high)          AS high,
            min(low)           AS low,
            last(close, time)  AS close,
            sum(volume)        AS volume
        FROM tick_data
        WHERE symbol = :symbol AND time >= :since
        GROUP BY ts
        ORDER BY ts ASC
    """)
    result = await db.execute(query, {"symbol": symbol.upper(), "since": since})
    rows   = result.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    return df
