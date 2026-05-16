"""
Sentiment & AI Signal Pipeline
- FinBERT for financial sentiment classification
- Reddit (r/IndiaInvestments, r/Nifty50) + Twitter scraping
- News aggregation (Moneycontrol, ET Markets RSS)
- Claude/GPT-4 as the final signal agent
"""

import asyncio
import httpx
import feedparser
from datetime import datetime, timedelta
from typing import Optional
import json

# FinBERT via HuggingFace Transformers
try:
    from transformers import pipeline
    _finbert = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        device=-1,         # CPU; change to 0 for GPU
        truncation=True,
        max_length=512,
    )
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False


LABEL_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


def finbert_score(texts: list[str]) -> list[dict]:
    """
    Returns list of {"label": ..., "score": float (-1 to 1)}
    Falls back to keyword heuristic if model unavailable.
    """
    if FINBERT_AVAILABLE:
        results = _finbert(texts, batch_size=16)
        return [
            {"label": r["label"], "score": LABEL_MAP.get(r["label"], 0) * r["score"]}
            for r in results
        ]
    # Simple keyword fallback
    bullish_kw = {"buy","bullish","breakout","rally","surge","growth","profit","beat"}
    bearish_kw = {"sell","bearish","crash","fall","decline","loss","miss","below","weak"}
    scores = []
    for text in texts:
        words = set(text.lower().split())
        bull_hits = len(words & bullish_kw)
        bear_hits = len(words & bearish_kw)
        if bull_hits > bear_hits:
            scores.append({"label": "positive", "score": min(bull_hits * 0.2, 1.0)})
        elif bear_hits > bull_hits:
            scores.append({"label": "negative", "score": -min(bear_hits * 0.2, 1.0)})
        else:
            scores.append({"label": "neutral", "score": 0.0})
    return scores


# ── Reddit scraper (no auth needed via pushshift mirror) ─────────────────────

class RedditScraper:
    SUBREDDITS = ["IndiaInvestments", "Nifty50", "IndianStockMarket", "DalalStreetTalks"]
    BASE_URL   = "https://www.reddit.com/r/{sub}/search.json"

    @classmethod
    async def fetch_mentions(cls, symbol: str, hours: int = 24) -> list[dict]:
        posts = []
        after = int((datetime.utcnow() - timedelta(hours=hours)).timestamp())
        async with httpx.AsyncClient(
            headers={"User-Agent": "IndiaTrader/1.0"}, timeout=10
        ) as client:
            for sub in cls.SUBREDDITS:
                try:
                    resp = await client.get(
                        cls.BASE_URL.format(sub=sub),
                        params={"q": symbol, "sort": "new", "limit": 50, "t": "day"},
                    )
                    if resp.status_code != 200:
                        continue
                    items = resp.json().get("data", {}).get("children", [])
                    for item in items:
                        d = item.get("data", {})
                        if d.get("created_utc", 0) < after:
                            continue
                        posts.append({
                            "source": f"reddit/{sub}",
                            "text":   (d.get("title", "") + " " + d.get("selftext", ""))[:512],
                            "score":  d.get("score", 0),
                            "url":    f"https://reddit.com{d.get('permalink','')}",
                            "time":   datetime.utcfromtimestamp(d.get("created_utc", 0)).isoformat(),
                        })
                except Exception:
                    continue
        return posts


# ── News RSS fetcher ─────────────────────────────────────────────────────────

class NewsAggregator:
    FEEDS = [
        "https://www.moneycontrol.com/rss/marketreports.xml",
        "https://economictimes.indiatimes.com/markets/stocks/rss.cms",
        "https://feeds.feedburner.com/ndtvprofit-latest",
    ]

    @classmethod
    async def fetch_headlines(cls, symbol: str, hours: int = 24) -> list[dict]:
        articles = []
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        async with httpx.AsyncClient(timeout=10) as client:
            for url in cls.FEEDS:
                try:
                    resp = await client.get(url)
                    feed = feedparser.parse(resp.text)
                    for entry in feed.entries:
                        title   = entry.get("title", "")
                        summary = entry.get("summary", "")
                        if symbol.upper() not in (title + summary).upper():
                            continue
                        pub = entry.get("published_parsed")
                        if pub:
                            pub_dt = datetime(*pub[:6])
                            if pub_dt < cutoff:
                                continue
                        articles.append({
                            "source": url.split("/")[2],
                            "text":   f"{title}. {summary}"[:512],
                            "url":    entry.get("link", ""),
                            "time":   pub_dt.isoformat() if pub else "",
                        })
                except Exception:
                    continue
        return articles


# ── Aggregated sentiment for a symbol ────────────────────────────────────────

async def compute_sentiment(symbol: str) -> dict:
    """
    Pulls Reddit + news, runs FinBERT, returns aggregated score.
    score: -1 (very bearish) to +1 (very bullish)
    """
    reddit_posts = await RedditScraper.fetch_mentions(symbol)
    news_items   = await NewsAggregator.fetch_headlines(symbol)

    all_items = reddit_posts + news_items
    if not all_items:
        return {"symbol": symbol, "score": 0.0, "magnitude": 0, "sources": []}

    texts  = [item["text"] for item in all_items]
    scores = finbert_score(texts)

    # Weighted average (Reddit score as weight for reddit, uniform for news)
    total_weight = 0.0
    weighted_sum = 0.0
    annotated    = []

    for item, s in zip(all_items, scores):
        weight = max(item.get("score", 1), 1) if "reddit" in item["source"] else 1
        weighted_sum  += s["score"] * weight
        total_weight  += abs(weight)
        annotated.append({**item, "sentiment": s["score"], "label": s["label"]})

    agg_score = weighted_sum / total_weight if total_weight else 0.0
    return {
        "symbol":    symbol,
        "score":     round(agg_score, 3),
        "magnitude": len(all_items),
        "sources":   annotated[:10],   # top 10 for display
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── Claude AI Signal Agent ────────────────────────────────────────────────────

class AISignalAgent:
    """
    Uses Anthropic Claude to synthesize technical + sentiment data
    into a structured BUY / SELL / SHORT / HOLD recommendation.
    """
    ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

    SYSTEM_PROMPT = """You are a professional Indian equity market analyst. 
    Given technical indicators, option chain data, and sentiment signals for a stock,
    produce a structured trading signal. Be concise, specific, and risk-aware.
    Always respond with valid JSON only."""

    @classmethod
    async def generate_signal(
        cls,
        symbol: str,
        ltp: float,
        technicals: dict,
        sentiment: dict,
        option_chain_summary: Optional[dict] = None,
        api_key: str = "",
    ) -> dict:
        """
        technicals: {rsi, macd, signal_line, bb_upper, bb_lower, atr, ema_20, ema_50}
        Returns structured signal dict.
        """
        prompt = f"""
Symbol: {symbol}
Current Price: ₹{ltp}

Technical Indicators:
- RSI(14): {technicals.get('rsi', 'N/A')}
- MACD: {technicals.get('macd', 'N/A')}, Signal: {technicals.get('signal_line', 'N/A')}
- Bollinger Bands: Upper={technicals.get('bb_upper','N/A')}, Lower={technicals.get('bb_lower','N/A')}
- ATR(14): {technicals.get('atr', 'N/A')}
- EMA(20): {technicals.get('ema_20','N/A')}, EMA(50): {technicals.get('ema_50','N/A')}

Sentiment Score: {sentiment.get('score', 0)} (scale -1 to +1, {sentiment.get('magnitude',0)} data points)
Top news/social context: {json.dumps(sentiment.get('sources',[])[:3], indent=2)}

Option Chain Summary: {json.dumps(option_chain_summary or {}, indent=2)}

Produce a JSON signal with this exact schema:
{{
  "action": "BUY|SELL|SHORT|HOLD",
  "confidence": 0.0-1.0,
  "target_price": float_or_null,
  "stop_loss": float_or_null,
  "time_horizon": "intraday|swing|positional",
  "rationale": "2-3 sentence explanation",
  "risk_factors": ["list", "of", "risks"],
  "suggested_strategy": "e.g. Bull Call Spread, Covered Call, Buy Stock, etc."
}}
"""
        if not api_key:
            # Return mock signal for development
            return cls._mock_signal(symbol, ltp, technicals, sentiment)

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                cls.ANTHROPIC_URL,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 500,
                    "system": cls.SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            resp.raise_for_status()
            content = resp.json()["content"][0]["text"]
            # Strip markdown code fences if present
            content = content.strip().strip("```json").strip("```").strip()
            signal = json.loads(content)
            signal["symbol"]     = symbol
            signal["created_at"] = datetime.utcnow().isoformat()
            return signal

    @staticmethod
    def _mock_signal(symbol, ltp, technicals, sentiment) -> dict:
        """Dev fallback — rule-based mock"""
        rsi = technicals.get("rsi", 50)
        sent = sentiment.get("score", 0)
        if rsi < 35 and sent > 0.1:
            action, conf = "BUY", 0.72
        elif rsi > 70 and sent < -0.1:
            action, conf = "SELL", 0.68
        elif rsi > 75:
            action, conf = "SHORT", 0.55
        else:
            action, conf = "HOLD", 0.40
        return {
            "symbol":   symbol, "action": action, "confidence": conf,
            "target_price": round(ltp * 1.05, 2) if action == "BUY" else None,
            "stop_loss":    round(ltp * 0.97, 2) if action in ("BUY","SHORT") else None,
            "time_horizon": "swing",
            "rationale": f"RSI at {rsi:.1f} with sentiment score {sent:.2f} suggests {action.lower()} setup.",
            "risk_factors": ["Market volatility", "Sector rotation risk"],
            "suggested_strategy": "Buy Stock" if action == "BUY" else "Exit / Hedge",
            "created_at": datetime.utcnow().isoformat(),
        }
