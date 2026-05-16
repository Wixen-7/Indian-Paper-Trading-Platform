# IndiaTrader — Indian Paper Trading Platform
# Full-stack setup guide

---

## Project Structure

```
trading_platform/
├── backend/
│   ├── main.py                        # FastAPI app entry point
│   ├── models.py                      # SQLAlchemy models
│   ├── database.py                    # DB session + engine
│   ├── requirements.txt
│   ├── routers/
│   │   └── all_routers.py             # trades, portfolio, options, signals, backtest
│   └── services/
│       ├── market_data.py             # NSE tick ingestion + Yahoo Finance
│       ├── options_engine.py          # Black-Scholes, Greeks, payoff
│       ├── sentiment_pipeline.py      # FinBERT + Reddit + News + Claude AI
│       ├── technical_indicators.py    # RSI, MACD, BB, ATR, EMA, VWAP
│       └── backtester.py              # Backtester + built-in + custom strategies
└── frontend/
    ├── package.json
    └── app/
        ├── dashboard/page.tsx         # Main dashboard + live chart + orders
        ├── options/page.tsx           # Option chain + payoff builder
        └── strategy/page.tsx          # Strategy studio + backtester
```

---

## Tech Stack

| Layer        | Technology                                                     |
|--------------|----------------------------------------------------------------|
| Backend      | FastAPI, SQLAlchemy (async), Celery, Redis                     |
| Database     | PostgreSQL + TimescaleDB (time-series ticks)                   |
| ML / NLP     | FinBERT (ProsusAI/finbert via HuggingFace Transformers)       |
| AI Signals   | Anthropic Claude claude-sonnet-4-20250514 via API              |
| Market Data  | Yahoo Finance (free), Zerodha Kite API, Angel One SmartAPI    |
| Options Math | scipy (Black-Scholes), numpy                                   |
| Frontend     | Next.js 14 (App Router), TypeScript, Tailwind CSS              |
| Charts       | TradingView Lightweight Charts (price), Recharts (analytics)  |
| Infra        | Docker Compose                                                 |

---

## Prerequisites

- Docker + Docker Compose
- Node.js 20+
- Python 3.11+
- Zerodha Kite API key (optional — Yahoo Finance works without it)
- Anthropic API key (for AI signals)

---

## Quick Start

### 1. Clone and configure

```bash
git clone <your-repo>
cd trading_platform
cp .env.example .env
# Edit .env with your API keys
```

### 2. Start infrastructure

```bash
docker-compose up -d
```

### 3. Install backend dependencies

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Initialize database

```bash
# Run migrations
alembic upgrade head

# Enable TimescaleDB extension (run once in psql)
# docker exec -it postgres psql -U trader -d indiatrader -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"

# Convert tick_data to hypertable
# SELECT create_hypertable('tick_data', 'time', if_not_exists => TRUE);

# Seed 2 years of historical data (takes ~5 minutes)
python -c "
import asyncio
from database import get_db_session
from services.market_data import YahooFinanceFetcher, MarketDataService
async def seed():
    async with get_db_session() as db:
        await YahooFinanceFetcher.seed_database(
            MarketDataService.NIFTY50_SYMBOLS + MarketDataService.INDEX_SYMBOLS,
            db, period='2y'
        )
asyncio.run(seed())
"
```

### 5. Start backend

```bash
uvicorn main:app --reload --port 8000
```

### 6. Start frontend

```bash
cd ../frontend
npm install
npm run dev
```

Open http://localhost:3000

---

## Environment Variables (.env)

```env
# Database
DATABASE_URL=postgresql+asyncpg://trader:trader@localhost:5432/indiatrader

# Redis
REDIS_URL=redis://localhost:6379

# Zerodha Kite (optional - for live WebSocket ticks)
KITE_API_KEY=your_api_key
KITE_ACCESS_TOKEN=your_access_token

# Angel One SmartAPI (optional fallback)
ANGEL_API_KEY=your_api_key
ANGEL_CLIENT_ID=your_client_id
ANGEL_PASSWORD=your_password
ANGEL_TOTP_SECRET=your_totp_secret

# Anthropic (for AI signals)
ANTHROPIC_API_KEY=sk-ant-...

# Twitter/X API v2 (optional - for social sentiment)
TWITTER_BEARER_TOKEN=your_bearer_token
```

---

## docker-compose.yml

```yaml
version: "3.9"
services:
  postgres:
    image: timescale/timescaledb:latest-pg16
    container_name: postgres
    environment:
      POSTGRES_DB: indiatrader
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: trader
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    container_name: redis
    ports:
      - "6379:6379"

  celery:
    build: ./backend
    command: celery -A tasks worker --loglevel=info -Q default,signals,backtest
    environment:
      - DATABASE_URL=postgresql+asyncpg://trader:trader@postgres:5432/indiatrader
      - REDIS_URL=redis://redis:6379
    depends_on: [postgres, redis]

volumes:
  pgdata:
```

---

## requirements.txt

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
sqlalchemy[asyncio]==2.0.35
asyncpg==0.29.0
alembic==1.13.3
redis[asyncio]==5.0.8
celery==5.4.0
httpx==0.27.2
pandas==2.2.3
numpy==2.1.2
scipy==1.14.1
pandas-ta==0.3.14b0
transformers==4.44.2
torch==2.4.1          # CPU-only: torch==2.4.1+cpu
feedparser==6.0.11
python-dotenv==1.0.1
pytz==2024.2
pydantic==2.9.2
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.12
# Optional broker integrations
kiteconnect==5.0.1
smartapi-python==1.5.1
pyotp==2.9.0
```

---

## package.json (frontend)

```json
{
  "name": "indiatrader-frontend",
  "version": "1.0.0",
  "scripts": {
    "dev":   "next dev",
    "build": "next build",
    "start": "next start"
  },
  "dependencies": {
    "next":                    "14.2.14",
    "react":                   "^18.3.1",
    "react-dom":               "^18.3.1",
    "typescript":              "^5.6.3",
    "tailwindcss":             "^3.4.13",
    "lightweight-charts":      "^4.2.0",
    "recharts":                "^2.13.0",
    "@types/react":            "^18.3.11",
    "@types/node":             "^22.7.4"
  }
}
```

---

## API Reference

### Trades
| Method | Endpoint                       | Description             |
|--------|--------------------------------|-------------------------|
| POST   | /api/trades/place              | Place paper order       |
| DELETE | /api/trades/{id}/cancel        | Cancel pending order    |
| GET    | /api/trades/history/{pid}      | Order history           |

### Portfolio
| Method | Endpoint                         | Description             |
|--------|----------------------------------|-------------------------|
| GET    | /api/portfolio/{id}/summary      | Live P&L + positions    |

### Options
| Method | Endpoint              | Description                  |
|--------|-----------------------|------------------------------|
| GET    | /api/options/chain/{symbol} | Live option chain       |
| POST   | /api/options/greeks   | Calculate Greeks + IV        |
| POST   | /api/options/payoff   | Multi-leg payoff diagram     |
| POST   | /api/options/margin   | Futures margin calc          |

### Signals
| Method | Endpoint                | Description                     |
|--------|-------------------------|---------------------------------|
| GET    | /api/signals/sentiment/{sym} | FinBERT sentiment score    |
| POST   | /api/signals/ai/{sym}   | AI signal (Claude agent)        |
| GET    | /api/signals/feed       | Recent signals feed             |

### Backtest
| Method | Endpoint                 | Description                   |
|--------|--------------------------|-------------------------------|
| POST   | /api/backtest/run        | Run backtest                  |
| GET    | /api/backtest/strategies | List available strategies     |

### WebSocket
| Endpoint                | Description                   |
|-------------------------|-------------------------------|
| /ws/ticks/{symbol}      | Live tick stream              |
| /ws/pnl/{user_id}       | Live portfolio P&L            |

---

## Resume Talking Points

When presenting this project in interviews, highlight:

1. **Real-time data pipeline** — WebSocket tick streaming via Redis pub/sub,
   TimescaleDB hypertable partitioning for time-series at scale.

2. **Options pricing from scratch** — Black-Scholes implementation with
   Newton-Raphson / Brent IV solver, full Greeks (Delta, Gamma, Theta, Vega, Rho).

3. **NLP pipeline** — FinBERT fine-tuned on financial text, aggregated
   across Reddit + news RSS with engagement-weighted sentiment scoring.

4. **AI agent architecture** — Claude used as a reasoning layer over
   structured technical + sentiment data (RAG-style context injection).

5. **Strategy framework** — Vectorized backtester with Sharpe, Sortino,
   drawdown metrics; no-code rule engine that compiles JSON → signal functions.

6. **System design** — Async FastAPI + Celery workers + Redis pub/sub;
   designed for horizontal scaling (each service independently deployable).
```
