"""
Indian Paper Trading Platform - Backend Core
Stack: FastAPI + PostgreSQL (TimescaleDB) + Redis + Celery
"""

from fastapi import FastAPI, WebSocket, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio, json, redis.asyncio as aioredis
from datetime import datetime, date
from typing import Optional
import httpx

from database import get_db, engine
from models import Base
from routers import trades, portfolio, options, signals, backtest
from services.market_data import MarketDataService
from services.options_engine import OptionsEngine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background data ingestion tasks
    asyncio.create_task(MarketDataService.stream_nse_ticks())
    asyncio.create_task(MarketDataService.ingest_option_chain())
    yield

app = FastAPI(
    title="IndiaTrader Paper Platform",
    description="Simulated trading with NSE/BSE data, F&O, and AI signals",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(trades.router,     prefix="/api/trades",     tags=["trades"])
app.include_router(portfolio.router,  prefix="/api/portfolio",  tags=["portfolio"])
app.include_router(options.router,    prefix="/api/options",    tags=["options"])
app.include_router(signals.router,    prefix="/api/signals",    tags=["signals"])
app.include_router(backtest.router,   prefix="/api/backtest",   tags=["backtest"])


# ── WebSocket: real-time tick streaming ──────────────────────────────────────
@app.websocket("/ws/ticks/{symbol}")
async def websocket_ticks(websocket: WebSocket, symbol: str):
    await websocket.accept()
    redis = aioredis.from_url("redis://localhost:6379")
    pubsub = redis.pubsub()
    await pubsub.subscribe(f"tick:{symbol.upper()}")
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                await websocket.send_text(message["data"].decode())
    except Exception:
        await websocket.close()
    finally:
        await pubsub.unsubscribe(f"tick:{symbol.upper()}")


# ── WebSocket: portfolio P&L live feed ───────────────────────────────────────
@app.websocket("/ws/pnl/{user_id}")
async def websocket_pnl(websocket: WebSocket, user_id: int):
    await websocket.accept()
    redis = aioredis.from_url("redis://localhost:6379")
    pubsub = redis.pubsub()
    await pubsub.subscribe(f"pnl:{user_id}")
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                await websocket.send_text(message["data"].decode())
    except Exception:
        await websocket.close()
