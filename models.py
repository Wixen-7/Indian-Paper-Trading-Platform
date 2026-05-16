"""
Database Models - TimescaleDB hypertables for tick data, standard tables for orders/portfolio
"""

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Date,
    ForeignKey, Enum, JSON, Text, UniqueConstraint
)
from sqlalchemy.orm import relationship, DeclarativeBase
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
import enum


class Base(DeclarativeBase):
    pass


class OrderSide(str, enum.Enum):
    BUY  = "BUY"
    SELL = "SELL"

class OrderType(str, enum.Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"
    SL     = "SL"
    SL_M   = "SL-M"

class OrderStatus(str, enum.Enum):
    PENDING   = "PENDING"
    EXECUTED  = "EXECUTED"
    CANCELLED = "CANCELLED"
    REJECTED  = "REJECTED"

class InstrumentType(str, enum.Enum):
    EQ      = "EQ"       # Equity
    FUT     = "FUT"      # Futures
    CE      = "CE"       # Call Option
    PE      = "PE"       # Put Option


class User(Base):
    __tablename__ = "users"
    id             = Column(Integer, primary_key=True)
    email          = Column(String, unique=True, nullable=False)
    username       = Column(String, unique=True, nullable=False)
    hashed_password= Column(String, nullable=False)
    created_at     = Column(DateTime, default=datetime.utcnow)
    portfolios     = relationship("Portfolio", back_populates="user")


class Portfolio(Base):
    __tablename__ = "portfolios"
    id             = Column(Integer, primary_key=True)
    user_id        = Column(Integer, ForeignKey("users.id"), nullable=False)
    name           = Column(String, default="Default")
    cash_balance   = Column(Float, default=1_000_000.0)   # ₹10L virtual cash
    created_at     = Column(DateTime, default=datetime.utcnow)
    user           = relationship("User", back_populates="portfolios")
    positions      = relationship("Position", back_populates="portfolio")
    orders         = relationship("Order", back_populates="portfolio")


class Order(Base):
    __tablename__ = "orders"
    id             = Column(Integer, primary_key=True)
    portfolio_id   = Column(Integer, ForeignKey("portfolios.id"))
    symbol         = Column(String, nullable=False)         # e.g. RELIANCE, NIFTY24DECFUT
    instrument_type= Column(Enum(InstrumentType), default=InstrumentType.EQ)
    side           = Column(Enum(OrderSide), nullable=False)
    order_type     = Column(Enum(OrderType), default=OrderType.MARKET)
    quantity       = Column(Integer, nullable=False)
    limit_price    = Column(Float, nullable=True)
    trigger_price  = Column(Float, nullable=True)
    executed_price = Column(Float, nullable=True)
    status         = Column(Enum(OrderStatus), default=OrderStatus.PENDING)
    # F&O specific
    expiry_date    = Column(Date, nullable=True)
    strike_price   = Column(Float, nullable=True)
    option_type    = Column(String, nullable=True)          # CE / PE
    lot_size       = Column(Integer, default=1)
    created_at     = Column(DateTime, default=datetime.utcnow)
    executed_at    = Column(DateTime, nullable=True)
    portfolio      = relationship("Portfolio", back_populates="orders")


class Position(Base):
    __tablename__ = "positions"
    id             = Column(Integer, primary_key=True)
    portfolio_id   = Column(Integer, ForeignKey("portfolios.id"))
    symbol         = Column(String, nullable=False)
    instrument_type= Column(Enum(InstrumentType), default=InstrumentType.EQ)
    quantity       = Column(Integer, default=0)             # negative = short
    avg_price      = Column(Float, nullable=False)
    # F&O
    expiry_date    = Column(Date, nullable=True)
    strike_price   = Column(Float, nullable=True)
    option_type    = Column(String, nullable=True)
    lot_size       = Column(Integer, default=1)
    # Greeks (updated periodically)
    delta          = Column(Float, nullable=True)
    gamma          = Column(Float, nullable=True)
    theta          = Column(Float, nullable=True)
    vega           = Column(Float, nullable=True)
    iv             = Column(Float, nullable=True)
    portfolio      = relationship("Portfolio", back_populates="positions")

    __table_args__ = (
        UniqueConstraint("portfolio_id", "symbol", "expiry_date", "strike_price", "option_type"),
    )


class TickData(Base):
    """TimescaleDB hypertable — partition by time"""
    __tablename__ = "tick_data"
    time           = Column(DateTime, primary_key=True)
    symbol         = Column(String, primary_key=True)
    ltp            = Column(Float)
    open           = Column(Float)
    high           = Column(Float)
    low            = Column(Float)
    close          = Column(Float)
    volume         = Column(Integer)
    oi             = Column(Integer, nullable=True)   # open interest for F&O


class SentimentSignal(Base):
    __tablename__ = "sentiment_signals"
    id             = Column(Integer, primary_key=True)
    symbol         = Column(String, nullable=False)
    source         = Column(String)                   # twitter / reddit / news
    score          = Column(Float)                    # -1 (bearish) to +1 (bullish)
    magnitude      = Column(Float)                    # volume/intensity
    summary        = Column(Text)
    raw_data       = Column(JSONB)
    created_at     = Column(DateTime, default=datetime.utcnow)


class AISignal(Base):
    __tablename__ = "ai_signals"
    id             = Column(Integer, primary_key=True)
    symbol         = Column(String, nullable=False)
    action         = Column(String)                   # BUY / SELL / SHORT / HOLD
    confidence     = Column(Float)                    # 0-1
    target_price   = Column(Float, nullable=True)
    stop_loss      = Column(Float, nullable=True)
    rationale      = Column(Text)
    sources        = Column(JSONB)                    # list of contributing signals
    created_at     = Column(DateTime, default=datetime.utcnow)
    valid_until    = Column(DateTime, nullable=True)
