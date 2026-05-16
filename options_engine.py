"""
Options & Futures Engine
- Black-Scholes pricing for CE/PE
- Greeks: Delta, Gamma, Theta, Vega, Rho
- Implied Volatility solver (Newton-Raphson)
- NSE option chain ingestion
- Futures P&L with margin calculation
"""

import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, date
from dataclasses import dataclass
from typing import Optional
import httpx


@dataclass
class OptionContract:
    symbol: str
    underlying_price: float
    strike: float
    expiry: date
    option_type: str        # "CE" or "PE"
    risk_free_rate: float = 0.065   # RBI repo rate ~6.5%
    dividend_yield: float = 0.01


@dataclass
class OptionPricing:
    theoretical_price: float
    delta: float
    gamma: float
    theta: float    # per day
    vega: float     # per 1% IV move
    rho: float
    iv: Optional[float] = None


class BlackScholes:

    @staticmethod
    def time_to_expiry(expiry: date) -> float:
        """Returns T in years"""
        today = datetime.utcnow().date()
        calendar_days = (expiry - today).days
        # NSE uses trading days (252/year) but we'll use calendar for simplicity
        return max(calendar_days / 365.0, 1e-6)

    @staticmethod
    def d1_d2(S, K, T, r, sigma, q=0.0):
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2

    @classmethod
    def price(cls, contract: OptionContract, sigma: float) -> OptionPricing:
        S = contract.underlying_price
        K = contract.strike
        T = cls.time_to_expiry(contract.expiry)
        r = contract.risk_free_rate
        q = contract.dividend_yield
        opt = contract.option_type.upper()

        d1, d2 = cls.d1_d2(S, K, T, r, sigma, q)

        # Price
        if opt == "CE":
            price = (S * math.exp(-q * T) * norm.cdf(d1)
                     - K * math.exp(-r * T) * norm.cdf(d2))
        else:
            price = (K * math.exp(-r * T) * norm.cdf(-d2)
                     - S * math.exp(-q * T) * norm.cdf(-d1))

        # Greeks
        phi = norm.pdf(d1)

        # Delta
        if opt == "CE":
            delta = math.exp(-q * T) * norm.cdf(d1)
        else:
            delta = -math.exp(-q * T) * norm.cdf(-d1)

        # Gamma (same for CE and PE)
        gamma = (math.exp(-q * T) * phi) / (S * sigma * math.sqrt(T))

        # Theta (per day)
        theta_common = -(S * math.exp(-q * T) * phi * sigma) / (2 * math.sqrt(T))
        if opt == "CE":
            theta = (theta_common
                     - r * K * math.exp(-r * T) * norm.cdf(d2)
                     + q * S * math.exp(-q * T) * norm.cdf(d1)) / 365
        else:
            theta = (theta_common
                     + r * K * math.exp(-r * T) * norm.cdf(-d2)
                     - q * S * math.exp(-q * T) * norm.cdf(-d1)) / 365

        # Vega (per 1% IV move)
        vega = S * math.exp(-q * T) * phi * math.sqrt(T) / 100

        # Rho (per 1% rate move)
        if opt == "CE":
            rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100

        return OptionPricing(
            theoretical_price=round(price, 2),
            delta=round(delta, 4),
            gamma=round(gamma, 6),
            theta=round(theta, 4),
            vega=round(vega, 4),
            rho=round(rho, 4),
        )

    @classmethod
    def implied_volatility(
        cls, contract: OptionContract, market_price: float
    ) -> Optional[float]:
        """Solve IV using Brent's method (more robust than Newton-Raphson)"""
        S = contract.underlying_price
        K = contract.strike
        T = cls.time_to_expiry(contract.expiry)
        r = contract.risk_free_rate
        q = contract.dividend_yield
        opt = contract.option_type.upper()

        def objective(sigma):
            pricing = cls.price(contract, sigma)
            return pricing.theoretical_price - market_price

        try:
            # Intrinsic value floor check
            if opt == "CE":
                intrinsic = max(S * math.exp(-q*T) - K * math.exp(-r*T), 0)
            else:
                intrinsic = max(K * math.exp(-r*T) - S * math.exp(-q*T), 0)

            if market_price < intrinsic - 0.01:
                return None  # Arbitrage / bad data

            iv = brentq(objective, 1e-4, 5.0, xtol=1e-5, maxiter=200)
            return round(iv, 4)
        except (ValueError, RuntimeError):
            return None

    @classmethod
    def price_with_iv(cls, contract: OptionContract, market_price: float) -> OptionPricing:
        """Price with market-implied IV"""
        iv = cls.implied_volatility(contract, market_price)
        if iv is None:
            iv = 0.20  # fallback 20% IV
        pricing = cls.price(contract, iv)
        pricing.iv = iv
        return pricing


class FuturesEngine:
    """
    NSE Futures P&L and margin calculation.
    Margin approx = SPAN + Exposure (simplified)
    """
    NIFTY_LOT_SIZE  = 50
    BANKNIFTY_LOT_SIZE = 15
    FINNIFTY_LOT_SIZE = 40

    @staticmethod
    def lot_size(symbol: str) -> int:
        symbol = symbol.upper()
        if "BANKNIFTY" in symbol:
            return FuturesEngine.BANKNIFTY_LOT_SIZE
        if "FINNIFTY" in symbol:
            return FuturesEngine.FINNIFTY_LOT_SIZE
        if "NIFTY" in symbol:
            return FuturesEngine.NIFTY_LOT_SIZE
        return 1  # equity futures — varies

    @staticmethod
    def margin_required(futures_price: float, lot_size: int, lots: int = 1) -> float:
        """Simplified SPAN margin ≈ 10% of contract value"""
        contract_value = futures_price * lot_size * lots
        span_margin = contract_value * 0.10
        exposure_margin = contract_value * 0.03
        return round(span_margin + exposure_margin, 2)

    @staticmethod
    def unrealised_pnl(
        entry_price: float,
        current_price: float,
        quantity: int,     # positive = long, negative = short
        lot_size: int,
    ) -> float:
        return round((current_price - entry_price) * quantity * lot_size, 2)


class OptionChainFetcher:
    """
    Fetches live option chain from NSE India (public endpoint).
    Returns structured data for the strike ladder UI.
    """
    NSE_OC_URL = "https://www.nseindia.com/api/option-chain-indices"
    HEADERS = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/option-chain",
    }

    @classmethod
    async def fetch(cls, symbol: str = "NIFTY") -> dict:
        """
        symbol: NIFTY | BANKNIFTY | FINNIFTY
        Returns parsed OC data with calls and puts for all strikes.
        """
        async with httpx.AsyncClient(headers=cls.HEADERS, timeout=10.0) as client:
            # First hit the homepage to get session cookies
            await client.get("https://www.nseindia.com", timeout=5)
            resp = await client.get(
                cls.NSE_OC_URL, params={"symbol": symbol}
            )
            resp.raise_for_status()
            raw = resp.json()

        records = raw.get("records", {})
        data    = records.get("data", [])
        expiry_dates = records.get("expiryDates", [])
        underlying   = records.get("underlyingValue", 0)
        timestamp    = records.get("timestamp", "")

        # Parse into strike ladder
        strikes = {}
        for row in data:
            strike = row.get("strikePrice")
            expiry = row.get("expiryDate")
            if strike not in strikes:
                strikes[strike] = {}
            entry = {"expiry": expiry, "strike": strike}
            if "CE" in row:
                entry["CE"] = row["CE"]
            if "PE" in row:
                entry["PE"] = row["PE"]
            strikes[strike][expiry] = entry

        return {
            "symbol": symbol,
            "underlying": underlying,
            "timestamp": timestamp,
            "expiry_dates": expiry_dates,
            "strikes": strikes,
        }


class PayoffCalculator:
    """
    Calculate P&L payoff diagrams for single legs and multi-leg strategies.
    Used for the frontend payoff chart.
    """

    @staticmethod
    def payoff(
        legs: list[dict],   # [{type, strike, premium, qty, position}]
        price_range: tuple = None,
        underlying: float = None,
        num_points: int = 100,
    ) -> dict:
        """
        legs example:
          [{"type":"CE","strike":24000,"premium":150,"qty":1,"position":"long"},
           {"type":"PE","strike":23500,"premium":100,"qty":1,"position":"long"}]
        position: "long" | "short"
        Returns x (prices) and y (P&L) arrays.
        """
        if underlying is None:
            underlying = legs[0]["strike"]

        if price_range is None:
            lo = min(l["strike"] for l in legs) * 0.85
            hi = max(l["strike"] for l in legs) * 1.15
        else:
            lo, hi = price_range

        prices = np.linspace(lo, hi, num_points)
        total_pnl = np.zeros(num_points)

        for leg in legs:
            strike  = leg["strike"]
            premium = leg["premium"]
            qty     = leg["qty"]
            pos_sign = 1 if leg["position"] == "long" else -1
            t = leg["type"].upper()

            if t == "CE":
                intrinsic = np.maximum(prices - strike, 0)
            elif t == "PE":
                intrinsic = np.maximum(strike - prices, 0)
            else:   # FUT
                intrinsic = prices - strike

            pnl = pos_sign * (intrinsic - premium) * qty
            total_pnl += pnl

        # Find breakevens (zero crossings)
        breakevens = []
        for i in range(len(total_pnl) - 1):
            if total_pnl[i] * total_pnl[i+1] < 0:
                be = prices[i] - total_pnl[i] * (prices[i+1] - prices[i]) / (total_pnl[i+1] - total_pnl[i])
                breakevens.append(round(float(be), 2))

        return {
            "prices":     [round(float(p), 2) for p in prices],
            "pnl":        [round(float(p), 2) for p in total_pnl],
            "max_profit": round(float(total_pnl.max()), 2),
            "max_loss":   round(float(total_pnl.min()), 2),
            "breakevens": breakevens,
        }
