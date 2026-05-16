"""
Strategy Backtester
- Built-in strategies: RSI mean reversion, MACD crossover, Bollinger Breakout,
  SuperTrend, EMA crossover
- Custom no-code rule engine (JSON strategy definition)
- Performance metrics: Sharpe, Sortino, Max Drawdown, Win Rate, CAGR
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional
from datetime import datetime
import json


# ── Performance metrics ───────────────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, risk_free: float = 0.065) -> float:
    excess = returns - risk_free / 252
    return round(float(excess.mean() / excess.std() * np.sqrt(252)), 3) if excess.std() > 0 else 0.0


def sortino_ratio(returns: pd.Series, risk_free: float = 0.065) -> float:
    excess    = returns - risk_free / 252
    downside  = excess[excess < 0].std()
    return round(float(excess.mean() / downside * np.sqrt(252)), 3) if downside > 0 else 0.0


def max_drawdown(equity_curve: pd.Series) -> float:
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return round(float(drawdown.min()), 4)


def cagr(equity_curve: pd.Series) -> float:
    n_years = len(equity_curve) / 252
    if n_years <= 0:
        return 0.0
    return round(float((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / n_years) - 1), 4)


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_date:  datetime
    exit_date:   Optional[datetime]
    symbol:      str
    side:        str            # "long" | "short"
    entry_price: float
    exit_price:  Optional[float]
    quantity:    int
    pnl:         float = 0.0
    pnl_pct:     float = 0.0
    exit_reason: str = ""       # "signal" | "stop_loss" | "target" | "end_of_data"


# ── Backtester core ───────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    strategy_name:  str
    symbol:         str
    start_date:     str
    end_date:       str
    initial_capital: float
    final_capital:  float
    total_return:   float
    cagr:           float
    sharpe:         float
    sortino:        float
    max_drawdown:   float
    win_rate:       float
    total_trades:   int
    avg_trade_pnl:  float
    best_trade:     float
    worst_trade:    float
    equity_curve:   list[float]  = field(default_factory=list)
    trades:         list[dict]   = field(default_factory=list)


class Backtester:

    def __init__(
        self,
        df: pd.DataFrame,
        symbol: str,
        initial_capital: float = 1_000_000,
        commission_pct: float = 0.0003,   # 0.03% per leg (Zerodha equity)
        slippage_pct: float = 0.0002,
    ):
        self.df              = df.copy()
        self.symbol          = symbol
        self.initial_capital = initial_capital
        self.commission      = commission_pct
        self.slippage        = slippage_pct

    def _execution_price(self, price: float, side: str) -> float:
        """Apply slippage: buys pay more, sells receive less."""
        if side == "buy":
            return price * (1 + self.slippage)
        return price * (1 - self.slippage)

    def run(
        self,
        strategy_fn: Callable[[pd.DataFrame], pd.Series],
        strategy_name: str = "Custom",
        position_size_pct: float = 0.95,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
    ) -> BacktestResult:
        """
        strategy_fn: receives df with indicators, returns pd.Series of signals:
            +1 = go long, -1 = go short / exit long, 0 = flat
        """
        df     = self.df.copy()
        signal = strategy_fn(df)
        df["signal"] = signal.reindex(df.index).fillna(0)

        capital   = self.initial_capital
        position  = 0          # shares held (+) or shorted (-)
        entry_px  = 0.0
        equity    = []
        trades    = []

        for i, (ts, row) in enumerate(df.iterrows()):
            price  = row["close"]
            sig    = int(row["signal"])

            # Stop-loss / take-profit check (uses today's low/high)
            if position != 0 and entry_px > 0:
                if position > 0:
                    if stop_loss_pct and row["low"] <= entry_px * (1 - stop_loss_pct):
                        exit_px = entry_px * (1 - stop_loss_pct)
                        pnl = (exit_px - entry_px) * position - abs(position) * exit_px * self.commission
                        capital  += position * exit_px + pnl
                        trades.append(Trade(
                            entry_date=None, exit_date=ts, symbol=self.symbol,
                            side="long", entry_price=entry_px, exit_price=exit_px,
                            quantity=position, pnl=round(pnl,2),
                            pnl_pct=round((exit_px/entry_px-1)*100,2),
                            exit_reason="stop_loss"
                        ))
                        position = 0; entry_px = 0.0
                    elif take_profit_pct and row["high"] >= entry_px * (1 + take_profit_pct):
                        exit_px = entry_px * (1 + take_profit_pct)
                        pnl = (exit_px - entry_px) * position - abs(position) * exit_px * self.commission
                        capital  += position * exit_px + pnl
                        trades.append(Trade(
                            entry_date=None, exit_date=ts, symbol=self.symbol,
                            side="long", entry_price=entry_px, exit_price=exit_px,
                            quantity=position, pnl=round(pnl,2),
                            pnl_pct=round((exit_px/entry_px-1)*100,2),
                            exit_reason="target"
                        ))
                        position = 0; entry_px = 0.0

            # Signal execution
            if sig == 1 and position <= 0:
                # Close short if any
                if position < 0:
                    exit_px  = self._execution_price(price, "buy")
                    pnl = (entry_px - exit_px) * abs(position) - abs(position) * exit_px * self.commission
                    capital += pnl
                    trades[-1].exit_date  = ts
                    trades[-1].exit_price = exit_px
                    trades[-1].pnl        = round(pnl, 2)
                    trades[-1].exit_reason = "signal"
                    position = 0

                # Go long
                buy_px   = self._execution_price(price, "buy")
                qty      = int((capital * position_size_pct) / buy_px)
                if qty > 0:
                    cost     = qty * buy_px * (1 + self.commission)
                    capital -= cost
                    position = qty
                    entry_px = buy_px
                    trades.append(Trade(
                        entry_date=ts, exit_date=None, symbol=self.symbol,
                        side="long", entry_price=buy_px, exit_price=None, quantity=qty
                    ))

            elif sig == -1 and position >= 0:
                # Close long if any
                if position > 0:
                    sell_px  = self._execution_price(price, "sell")
                    pnl = (sell_px - entry_px) * position - position * sell_px * self.commission
                    capital += position * sell_px
                    trades[-1].exit_date  = ts
                    trades[-1].exit_price = sell_px
                    trades[-1].pnl        = round(pnl, 2)
                    trades[-1].pnl_pct    = round((sell_px/entry_px-1)*100, 2)
                    trades[-1].exit_reason = "signal"
                    position = 0

            # Mark-to-market equity
            mtm = capital + position * price
            equity.append(mtm)

        # Close any open position at end
        if position != 0:
            last_px  = df["close"].iloc[-1]
            sell_px  = self._execution_price(last_px, "sell" if position > 0 else "buy")
            pnl = (sell_px - entry_px) * abs(position) * (1 if position > 0 else -1)
            capital += abs(position) * sell_px
            if trades:
                trades[-1].exit_date  = df.index[-1]
                trades[-1].exit_price = sell_px
                trades[-1].pnl        = round(pnl, 2)
                trades[-1].exit_reason = "end_of_data"
            equity[-1] = capital

        eq_series = pd.Series(equity)
        returns   = eq_series.pct_change().dropna()

        completed = [t for t in trades if t.exit_price is not None]
        pnls      = [t.pnl for t in completed]

        return BacktestResult(
            strategy_name   = strategy_name,
            symbol          = self.symbol,
            start_date      = str(df.index[0].date()),
            end_date        = str(df.index[-1].date()),
            initial_capital = self.initial_capital,
            final_capital   = round(equity[-1] if equity else self.initial_capital, 2),
            total_return    = round((equity[-1] / self.initial_capital - 1) * 100, 2) if equity else 0,
            cagr            = cagr(eq_series) * 100,
            sharpe          = sharpe_ratio(returns),
            sortino         = sortino_ratio(returns),
            max_drawdown    = max_drawdown(eq_series) * 100,
            win_rate        = round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1) if pnls else 0,
            total_trades    = len(completed),
            avg_trade_pnl   = round(np.mean(pnls), 2) if pnls else 0,
            best_trade      = round(max(pnls), 2) if pnls else 0,
            worst_trade     = round(min(pnls), 2) if pnls else 0,
            equity_curve    = [round(e, 2) for e in equity],
            trades          = [
                {
                    "entry_date":  str(t.entry_date)[:10] if t.entry_date else "",
                    "exit_date":   str(t.exit_date)[:10]  if t.exit_date  else "",
                    "side":        t.side,
                    "entry_price": t.entry_price,
                    "exit_price":  t.exit_price,
                    "quantity":    t.quantity,
                    "pnl":         t.pnl,
                    "pnl_pct":     t.pnl_pct,
                    "exit_reason": t.exit_reason,
                }
                for t in completed
            ],
        )


# ── Built-in strategies ───────────────────────────────────────────────────────

class BuiltInStrategies:

    @staticmethod
    def rsi_mean_reversion(df: pd.DataFrame, oversold: float = 30, overbought: float = 70) -> pd.Series:
        """Buy when RSI < oversold, sell when RSI > overbought."""
        signal = pd.Series(0, index=df.index)
        signal[df["rsi"] < oversold]  =  1
        signal[df["rsi"] > overbought] = -1
        return signal

    @staticmethod
    def macd_crossover(df: pd.DataFrame) -> pd.Series:
        """Buy on MACD crossing above signal line, sell on crossing below."""
        signal = pd.Series(0, index=df.index)
        cross_up   = (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
        cross_down = (df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))
        signal[cross_up]   =  1
        signal[cross_down] = -1
        return signal

    @staticmethod
    def ema_crossover(df: pd.DataFrame, fast: str = "ema_20", slow: str = "ema_50") -> pd.Series:
        """Golden cross / death cross."""
        signal = pd.Series(0, index=df.index)
        cross_up   = (df[fast] > df[slow]) & (df[fast].shift(1) <= df[slow].shift(1))
        cross_down = (df[fast] < df[slow]) & (df[fast].shift(1) >= df[slow].shift(1))
        signal[cross_up]   =  1
        signal[cross_down] = -1
        return signal

    @staticmethod
    def bollinger_breakout(df: pd.DataFrame) -> pd.Series:
        """Buy on close above upper band (momentum), sell on close below lower band."""
        signal = pd.Series(0, index=df.index)
        signal[df["close"] > df["bb_upper"]] =  1
        signal[df["close"] < df["bb_lower"]] = -1
        return signal

    @staticmethod
    def supertrend_follow(df: pd.DataFrame) -> pd.Series:
        """Follow SuperTrend direction changes."""
        if "supertrend_dir" not in df.columns:
            return pd.Series(0, index=df.index)
        signal = pd.Series(0, index=df.index)
        up   = (df["supertrend_dir"] ==  1) & (df["supertrend_dir"].shift(1) == -1)
        down = (df["supertrend_dir"] == -1) & (df["supertrend_dir"].shift(1) ==  1)
        signal[up]   =  1
        signal[down] = -1
        return signal

    @staticmethod
    def vwap_reversion(df: pd.DataFrame) -> pd.Series:
        """Intraday: buy below VWAP, sell above VWAP."""
        signal = pd.Series(0, index=df.index)
        signal[df["close"] < df["vwap"] * 0.995] =  1
        signal[df["close"] > df["vwap"] * 1.005] = -1
        return signal


STRATEGY_REGISTRY = {
    "rsi_mean_reversion":  BuiltInStrategies.rsi_mean_reversion,
    "macd_crossover":      BuiltInStrategies.macd_crossover,
    "ema_crossover":       BuiltInStrategies.ema_crossover,
    "bollinger_breakout":  BuiltInStrategies.bollinger_breakout,
    "supertrend_follow":   BuiltInStrategies.supertrend_follow,
    "vwap_reversion":      BuiltInStrategies.vwap_reversion,
}


# ── Custom no-code rule engine ────────────────────────────────────────────────

class RuleEngine:
    """
    Interprets a JSON strategy definition into a signal function.

    Example strategy JSON:
    {
      "name": "My RSI + EMA Strategy",
      "entry_long": [
        {"indicator": "rsi", "op": "<", "value": 35},
        {"indicator": "close", "op": ">", "indicator2": "ema_50"}
      ],
      "exit_long": [
        {"indicator": "rsi", "op": ">", "value": 65}
      ],
      "entry_short": [],
      "exit_short": []
    }
    Conditions within a list are AND-ed.
    """
    OPS = {
        ">":  lambda a, b: a > b,
        "<":  lambda a, b: a < b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
        "crossover":  lambda a, b: None,   # handled separately
        "crossunder": lambda a, b: None,
    }

    @classmethod
    def _evaluate_condition(cls, df: pd.DataFrame, cond: dict) -> pd.Series:
        ind1 = df[cond["indicator"]]
        op   = cond["op"]

        if "indicator2" in cond:
            ind2 = df[cond["indicator2"]]
        else:
            ind2 = pd.Series(cond["value"], index=df.index)

        if op == "crossover":
            return (ind1 > ind2) & (ind1.shift(1) <= ind2.shift(1))
        if op == "crossunder":
            return (ind1 < ind2) & (ind1.shift(1) >= ind2.shift(1))

        fn = cls.OPS.get(op, lambda a, b: pd.Series(False, index=df.index))
        return fn(ind1, ind2)

    @classmethod
    def build_signal_fn(cls, strategy_json: dict) -> Callable:
        def signal_fn(df: pd.DataFrame) -> pd.Series:
            signal = pd.Series(0, index=df.index)

            # Entry long: all conditions AND-ed
            if strategy_json.get("entry_long"):
                mask = pd.Series(True, index=df.index)
                for cond in strategy_json["entry_long"]:
                    mask &= cls._evaluate_condition(df, cond)
                signal[mask] = 1

            # Exit long / entry short
            if strategy_json.get("exit_long") or strategy_json.get("entry_short"):
                conditions = strategy_json.get("exit_long", []) + strategy_json.get("entry_short", [])
                mask = pd.Series(True, index=df.index)
                for cond in conditions:
                    mask &= cls._evaluate_condition(df, cond)
                signal[mask] = -1

            return signal
        return signal_fn
