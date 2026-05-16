"""
API Routers
- /api/trades    — place / cancel paper orders
- /api/portfolio — positions, P&L, summary
- /api/options   — option chain, payoff, Greeks
- /api/signals   — AI + sentiment signals
- /api/backtest  — run strategy backtests
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date, datetime
import json

from database import get_db
from models import (
    Order, Position, Portfolio, User,
    OrderSide, OrderType, OrderStatus, InstrumentType,
    AISignal, SentimentSignal
)
from services.options_engine import (
    BlackScholes, FuturesEngine, OptionContract,
    OptionChainFetcher, PayoffCalculator
)
from services.sentiment_pipeline import compute_sentiment, AISignalAgent
from services.technical_indicators import TechnicalIndicators, fetch_ohlcv
from services.backtester import Backtester, STRATEGY_REGISTRY, RuleEngine


# ─────────────────────────────────────────────────────────────────────────────
# TRADES ROUTER
# ─────────────────────────────────────────────────────────────────────────────

router_trades = APIRouter()


class PlaceOrderRequest(BaseModel):
    portfolio_id:    int
    symbol:          str
    instrument_type: InstrumentType = InstrumentType.EQ
    side:            OrderSide
    order_type:      OrderType = OrderType.MARKET
    quantity:        int       = Field(gt=0)
    limit_price:     Optional[float] = None
    trigger_price:   Optional[float] = None
    # F&O fields
    expiry_date:     Optional[date] = None
    strike_price:    Optional[float] = None
    option_type:     Optional[str]  = None   # CE | PE
    lot_size:        int = 1


class OrderMatchingService:
    """
    Simulates paper order execution.
    MARKET orders fill at LTP from Redis cache.
    LIMIT orders are queued and checked against incoming ticks.
    """
    @staticmethod
    async def get_ltp(symbol: str) -> float:
        import redis.asyncio as aioredis
        r = aioredis.from_url("redis://localhost:6379")
        val = await r.get(f"ltp:{symbol.upper()}")
        await r.aclose()
        if val:
            return float(val)
        raise HTTPException(status_code=404, detail=f"No market data for {symbol}")

    @classmethod
    async def execute_market_order(cls, order: Order, db: AsyncSession) -> float:
        ltp = await cls.get_ltp(order.symbol)
        order.executed_price = ltp
        order.status         = OrderStatus.EXECUTED
        order.executed_at    = datetime.utcnow()

        # Update or create position
        await cls._update_position(order, ltp, db)

        # Deduct / credit cash
        portfolio = await db.get(Portfolio, order.portfolio_id)
        cost = ltp * order.quantity * order.lot_size
        if order.side == OrderSide.BUY:
            if portfolio.cash_balance < cost:
                order.status = OrderStatus.REJECTED
                return ltp
            portfolio.cash_balance -= cost
        else:
            portfolio.cash_balance += cost

        await db.commit()
        return ltp

    @staticmethod
    async def _update_position(order: Order, price: float, db: AsyncSession):
        result = await db.execute(
            select(Position).where(
                Position.portfolio_id == order.portfolio_id,
                Position.symbol        == order.symbol,
                Position.expiry_date   == order.expiry_date,
                Position.strike_price  == order.strike_price,
                Position.option_type   == order.option_type,
            )
        )
        pos = result.scalar_one_or_none()
        qty_delta = order.quantity if order.side == OrderSide.BUY else -order.quantity

        if pos is None:
            pos = Position(
                portfolio_id   = order.portfolio_id,
                symbol         = order.symbol,
                instrument_type= order.instrument_type,
                quantity       = qty_delta,
                avg_price      = price,
                expiry_date    = order.expiry_date,
                strike_price   = order.strike_price,
                option_type    = order.option_type,
                lot_size       = order.lot_size,
            )
            db.add(pos)
        else:
            old_qty = pos.quantity
            new_qty = old_qty + qty_delta
            if new_qty == 0:
                await db.delete(pos)
                return
            if (old_qty > 0 and qty_delta > 0) or (old_qty < 0 and qty_delta < 0):
                pos.avg_price = (pos.avg_price * abs(old_qty) + price * abs(qty_delta)) / abs(new_qty)
            pos.quantity = new_qty


@router_trades.post("/place")
async def place_order(req: PlaceOrderRequest, db: AsyncSession = Depends(get_db)):
    order = Order(**req.model_dump())
    db.add(order)
    await db.flush()

    exec_price = None
    if req.order_type == OrderType.MARKET:
        exec_price = await OrderMatchingService.execute_market_order(order, db)
    else:
        await db.commit()

    return {
        "order_id":      order.id,
        "status":        order.status,
        "executed_price": exec_price,
        "message":       "Order placed successfully",
    }


@router_trades.delete("/{order_id}/cancel")
async def cancel_order(order_id: int, db: AsyncSession = Depends(get_db)):
    order = await db.get(Order, order_id)
    if not order:
        raise HTTPException(404, "Order not found")
    if order.status != OrderStatus.PENDING:
        raise HTTPException(400, f"Cannot cancel order in state {order.status}")
    order.status = OrderStatus.CANCELLED
    await db.commit()
    return {"message": "Order cancelled"}


@router_trades.get("/history/{portfolio_id}")
async def order_history(portfolio_id: int, limit: int = 50, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Order)
        .where(Order.portfolio_id == portfolio_id)
        .order_by(Order.created_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO ROUTER
# ─────────────────────────────────────────────────────────────────────────────

router_portfolio = APIRouter()


@router_portfolio.get("/{portfolio_id}/summary")
async def portfolio_summary(portfolio_id: int, db: AsyncSession = Depends(get_db)):
    portfolio = await db.get(Portfolio, portfolio_id)
    if not portfolio:
        raise HTTPException(404, "Portfolio not found")

    result = await db.execute(
        select(Position).where(Position.portfolio_id == portfolio_id)
    )
    positions = result.scalars().all()

    # Fetch LTPs from Redis
    import redis.asyncio as aioredis
    r = aioredis.from_url("redis://localhost:6379")
    total_invested = 0.0
    total_current  = 0.0
    pos_data       = []

    for p in positions:
        try:
            ltp = float(await r.get(f"ltp:{p.symbol}") or p.avg_price)
        except Exception:
            ltp = p.avg_price

        invested = p.avg_price * abs(p.quantity) * p.lot_size
        current  = ltp * abs(p.quantity) * p.lot_size
        unrealised = (ltp - p.avg_price) * p.quantity * p.lot_size

        total_invested += invested
        total_current  += current

        pos_data.append({
            "symbol":       p.symbol,
            "type":         p.instrument_type,
            "quantity":     p.quantity,
            "avg_price":    round(p.avg_price, 2),
            "ltp":          round(ltp, 2),
            "invested":     round(invested, 2),
            "current_value":round(current, 2),
            "unrealised_pnl": round(unrealised, 2),
            "unrealised_pct": round((ltp / p.avg_price - 1) * 100, 2),
            "delta":        p.delta,
            "theta":        p.theta,
        })

    await r.aclose()

    return {
        "portfolio_id":    portfolio_id,
        "name":            portfolio.name,
        "cash_balance":    round(portfolio.cash_balance, 2),
        "invested_value":  round(total_invested, 2),
        "current_value":   round(total_current, 2),
        "total_pnl":       round(total_current - total_invested, 2),
        "total_pnl_pct":   round((total_current / total_invested - 1) * 100, 2) if total_invested else 0,
        "positions":       pos_data,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONS ROUTER
# ─────────────────────────────────────────────────────────────────────────────

router_options = APIRouter()


@router_options.get("/chain/{symbol}")
async def option_chain(symbol: str = "NIFTY"):
    try:
        chain = await OptionChainFetcher.fetch(symbol)
        return chain
    except Exception as e:
        raise HTTPException(503, f"Could not fetch option chain: {e}")


class GreeksRequest(BaseModel):
    symbol:          str
    underlying:      float
    strike:          float
    expiry:          date
    option_type:     str    # CE | PE
    market_price:    Optional[float] = None
    sigma:           float = 0.20


@router_options.post("/greeks")
async def calculate_greeks(req: GreeksRequest):
    contract = OptionContract(
        symbol=req.symbol, underlying_price=req.underlying,
        strike=req.strike, expiry=req.expiry, option_type=req.option_type,
    )
    if req.market_price:
        pricing = BlackScholes.price_with_iv(contract, req.market_price)
    else:
        pricing = BlackScholes.price(contract, req.sigma)
    return pricing


class PayoffRequest(BaseModel):
    legs:       list[dict]
    underlying: Optional[float] = None


@router_options.post("/payoff")
async def payoff_diagram(req: PayoffRequest):
    result = PayoffCalculator.payoff(req.legs, underlying=req.underlying)
    return result


class MarginRequest(BaseModel):
    symbol:        str
    futures_price: float
    lots:          int = 1


@router_options.post("/margin")
async def futures_margin(req: MarginRequest):
    lot_size = FuturesEngine.lot_size(req.symbol)
    margin   = FuturesEngine.margin_required(req.futures_price, lot_size, req.lots)
    return {
        "symbol":          req.symbol,
        "lot_size":        lot_size,
        "lots":            req.lots,
        "contract_value":  round(req.futures_price * lot_size * req.lots, 2),
        "margin_required": margin,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIGNALS ROUTER
# ─────────────────────────────────────────────────────────────────────────────

router_signals = APIRouter()


@router_signals.get("/sentiment/{symbol}")
async def sentiment_signal(symbol: str):
    result = await compute_sentiment(symbol)
    return result


@router_signals.post("/ai/{symbol}")
async def ai_signal(
    symbol: str,
    db: AsyncSession = Depends(get_db),
    api_key: str = "",
):
    df = await fetch_ohlcv(symbol, db, days=200)
    if df.empty:
        raise HTTPException(404, f"No OHLCV data for {symbol}")

    df          = TechnicalIndicators.compute(df)
    technicals  = TechnicalIndicators.latest_snapshot(df)
    ltp         = float(df["close"].iloc[-1])
    sentiment   = await compute_sentiment(symbol)

    signal = await AISignalAgent.generate_signal(
        symbol=symbol, ltp=ltp,
        technicals=technicals, sentiment=sentiment,
        api_key=api_key,
    )

    # Persist signal
    db_signal = AISignal(
        symbol      = symbol,
        action      = signal.get("action"),
        confidence  = signal.get("confidence"),
        target_price= signal.get("target_price"),
        stop_loss   = signal.get("stop_loss"),
        rationale   = signal.get("rationale"),
        sources     = signal,
    )
    db.add(db_signal)
    await db.commit()

    return signal


@router_signals.get("/feed")
async def signals_feed(limit: int = 20, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(AISignal).order_by(AISignal.created_at.desc()).limit(limit)
    )
    return result.scalars().all()


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ROUTER
# ─────────────────────────────────────────────────────────────────────────────

router_backtest = APIRouter()


class BacktestRequest(BaseModel):
    symbol:            str
    strategy:          str = "rsi_mean_reversion"   # key from STRATEGY_REGISTRY
    custom_strategy:   Optional[dict] = None         # JSON rule engine definition
    days:              int = 365
    initial_capital:   float = 1_000_000
    position_size_pct: float = 0.95
    stop_loss_pct:     Optional[float] = None
    take_profit_pct:   Optional[float] = None


@router_backtest.post("/run")
async def run_backtest(req: BacktestRequest, db: AsyncSession = Depends(get_db)):
    df = await fetch_ohlcv(req.symbol, db, days=req.days)
    if df.empty:
        raise HTTPException(404, f"No data for {req.symbol}")

    df = TechnicalIndicators.compute(df)

    if req.custom_strategy:
        strategy_fn   = RuleEngine.build_signal_fn(req.custom_strategy)
        strategy_name = req.custom_strategy.get("name", "Custom")
    else:
        if req.strategy not in STRATEGY_REGISTRY:
            raise HTTPException(400, f"Unknown strategy '{req.strategy}'. Available: {list(STRATEGY_REGISTRY.keys())}")
        strategy_fn   = STRATEGY_REGISTRY[req.strategy]
        strategy_name = req.strategy

    bt = Backtester(
        df              = df,
        symbol          = req.symbol,
        initial_capital = req.initial_capital,
    )
    result = bt.run(
        strategy_fn        = strategy_fn,
        strategy_name      = strategy_name,
        position_size_pct  = req.position_size_pct,
        stop_loss_pct      = req.stop_loss_pct,
        take_profit_pct    = req.take_profit_pct,
    )
    return result


@router_backtest.get("/strategies")
async def list_strategies():
    return {
        "built_in": list(STRATEGY_REGISTRY.keys()),
        "custom_rule_operators": [">", "<", ">=", "<=", "==", "crossover", "crossunder"],
        "available_indicators": [
            "rsi", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_mid", "bb_lower",
            "ema_9", "ema_20", "ema_50", "ema_200",
            "atr", "vwap", "supertrend_dir",
            "stoch_k", "stoch_d", "obv",
            "open", "high", "low", "close", "volume",
        ],
    }
