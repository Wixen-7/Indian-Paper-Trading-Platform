/**
 * Main Dashboard - IndiaTrader Paper Platform
 * Stack: Next.js 14 (App Router) + Tailwind + TradingView Lightweight Charts
 * 
 * File: app/dashboard/page.tsx
 * 
 * Features:
 * - Live price chart (TradingView Lightweight Charts)
 * - Portfolio P&L summary cards
 * - AI signals feed
 * - Quick order placement
 * - Real-time WebSocket tick updates
 */

"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { createChart, ColorType, CrosshairMode, IChartApi, ISeriesApi } from "lightweight-charts";

// ─── Types ────────────────────────────────────────────────────────────────────

interface Tick {
  time: string;
  open: number; high: number; low: number; close: number; volume: number;
}

interface Position {
  symbol: string; type: string; quantity: number;
  avg_price: number; ltp: number; unrealised_pnl: number; unrealised_pct: number;
}

interface PortfolioSummary {
  cash_balance: number; invested_value: number;
  current_value: number; total_pnl: number; total_pnl_pct: number;
  positions: Position[];
}

interface Signal {
  id: number; symbol: string; action: string;
  confidence: number; target_price: number | null;
  stop_loss: number | null; rationale: string; created_at: string;
}

// ─── Chart Component ──────────────────────────────────────────────────────────

function PriceChart({ symbol }: { symbol: string }) {
  const chartRef  = useRef<HTMLDivElement>(null);
  const chart     = useRef<IChartApi | null>(null);
  const candleSeries = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeSeries = useRef<ISeriesApi<"Histogram"> | null>(null);

  useEffect(() => {
    if (!chartRef.current) return;

    chart.current = createChart(chartRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor:  "#888",
      },
      grid: {
        vertLines:  { color: "rgba(0,0,0,0.05)" },
        horzLines:  { color: "rgba(0,0,0,0.05)" },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: "rgba(0,0,0,0.1)" },
      timeScale:       { borderColor: "rgba(0,0,0,0.1)", timeVisible: true },
      width:  chartRef.current.clientWidth,
      height: 360,
    });

    candleSeries.current = chart.current.addCandlestickSeries({
      upColor:       "#22c55e",
      downColor:     "#ef4444",
      borderVisible: false,
      wickUpColor:   "#22c55e",
      wickDownColor: "#ef4444",
    });

    volumeSeries.current = chart.current.addHistogramSeries({
      color:     "rgba(59,130,246,0.3)",
      priceFormat:    { type: "volume" },
      priceScaleId:   "volume",
    });
    chart.current.priceScale("volume").applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });

    // Fetch historical data
    fetch(`/api/market/ohlcv/${symbol}?interval=1D&days=365`)
      .then(r => r.json())
      .then((data: Tick[]) => {
        const candles = data.map(d => ({
          time: d.time.split("T")[0] as any,
          open: d.open, high: d.high, low: d.low, close: d.close,
        }));
        const vols = data.map(d => ({
          time:  d.time.split("T")[0] as any,
          value: d.volume,
          color: d.close >= d.open ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)",
        }));
        candleSeries.current?.setData(candles);
        volumeSeries.current?.setData(vols);
        chart.current?.timeScale().fitContent();
      });

    const ro = new ResizeObserver(() => {
      if (chartRef.current)
        chart.current?.applyOptions({ width: chartRef.current.clientWidth });
    });
    ro.observe(chartRef.current);
    return () => { ro.disconnect(); chart.current?.remove(); };
  }, [symbol]);

  // Live tick via WebSocket
  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws/ticks/${symbol}`);
    ws.onmessage = (e) => {
      const tick = JSON.parse(e.data);
      const bar  = {
        time:  tick.time.split("T")[0] as any,
        open:  tick.open, high: tick.high, low: tick.low, close: tick.close,
      };
      candleSeries.current?.update(bar);
      volumeSeries.current?.update({ time: bar.time, value: tick.volume });
    };
    return () => ws.close();
  }, [symbol]);

  return <div ref={chartRef} className="w-full rounded-lg overflow-hidden" />;
}


// ─── Order Panel ──────────────────────────────────────────────────────────────

function OrderPanel({ symbol, portfolioId }: { symbol: string; portfolioId: number }) {
  const [side,      setSide]      = useState<"BUY" | "SELL">("BUY");
  const [orderType, setOrderType] = useState("MARKET");
  const [qty,       setQty]       = useState(1);
  const [price,     setPrice]     = useState("");
  const [instrument,setInstrument]= useState("EQ");
  const [loading,   setLoading]   = useState(false);
  const [message,   setMessage]   = useState("");

  const placeOrder = async () => {
    setLoading(true); setMessage("");
    try {
      const body: any = {
        portfolio_id: portfolioId,
        symbol, instrument_type: instrument,
        side, order_type: orderType, quantity: qty,
      };
      if (orderType === "LIMIT") body.limit_price = parseFloat(price);

      const resp = await fetch("/api/trades/place", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await resp.json();
      setMessage(data.status === "EXECUTED"
        ? `✓ Executed @ ₹${data.executed_price}`
        : `Order ${data.status}`);
    } catch {
      setMessage("❌ Order failed");
    }
    setLoading(false);
  };

  return (
    <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 rounded-xl p-5">
      <h3 className="text-sm font-medium text-zinc-500 mb-4">Place Order</h3>

      {/* Buy / Sell toggle */}
      <div className="flex rounded-lg overflow-hidden border border-zinc-200 dark:border-zinc-700 mb-4">
        {(["BUY","SELL"] as const).map(s => (
          <button key={s} onClick={() => setSide(s)}
            className={`flex-1 py-2 text-sm font-medium transition-colors
              ${side === s
                ? s === "BUY"
                  ? "bg-green-500 text-white"
                  : "bg-red-500 text-white"
                : "text-zinc-500 hover:bg-zinc-50 dark:hover:bg-zinc-800"}`}>
            {s}
          </button>
        ))}
      </div>

      {/* Instrument type */}
      <select value={instrument} onChange={e => setInstrument(e.target.value)}
        className="w-full mb-3 text-sm border border-zinc-200 dark:border-zinc-700 rounded-lg px-3 py-2 bg-transparent">
        <option value="EQ">Equity</option>
        <option value="FUT">Futures</option>
        <option value="CE">Call Option (CE)</option>
        <option value="PE">Put Option (PE)</option>
      </select>

      {/* Order type */}
      <select value={orderType} onChange={e => setOrderType(e.target.value)}
        className="w-full mb-3 text-sm border border-zinc-200 dark:border-zinc-700 rounded-lg px-3 py-2 bg-transparent">
        <option value="MARKET">Market</option>
        <option value="LIMIT">Limit</option>
        <option value="SL">Stop Loss</option>
      </select>

      {/* Quantity */}
      <input type="number" value={qty} onChange={e => setQty(+e.target.value)} min={1}
        placeholder="Quantity"
        className="w-full mb-3 text-sm border border-zinc-200 dark:border-zinc-700 rounded-lg px-3 py-2 bg-transparent" />

      {/* Limit price (conditional) */}
      {orderType !== "MARKET" && (
        <input type="number" value={price} onChange={e => setPrice(e.target.value)}
          placeholder="Price (₹)"
          className="w-full mb-3 text-sm border border-zinc-200 dark:border-zinc-700 rounded-lg px-3 py-2 bg-transparent" />
      )}

      <button onClick={placeOrder} disabled={loading}
        className={`w-full py-2.5 rounded-lg text-sm font-medium text-white transition-opacity
          ${side === "BUY" ? "bg-green-500 hover:bg-green-600" : "bg-red-500 hover:bg-red-600"}
          ${loading ? "opacity-50 cursor-not-allowed" : ""}`}>
        {loading ? "Placing..." : `${side} ${symbol}`}
      </button>

      {message && (
        <p className="mt-2 text-xs text-center text-zinc-500">{message}</p>
      )}
    </div>
  );
}


// ─── Signal Card ──────────────────────────────────────────────────────────────

function SignalCard({ signal }: { signal: Signal }) {
  const colors: Record<string, string> = {
    BUY:   "bg-green-50 text-green-700 border-green-200",
    SELL:  "bg-red-50  text-red-700  border-red-200",
    SHORT: "bg-orange-50 text-orange-700 border-orange-200",
    HOLD:  "bg-zinc-50  text-zinc-600 border-zinc-200",
  };
  const badge = colors[signal.action] ?? colors.HOLD;

  return (
    <div className="border border-zinc-100 dark:border-zinc-700 rounded-xl p-4 hover:border-zinc-300 transition-colors">
      <div className="flex items-center justify-between mb-2">
        <span className="font-medium text-sm">{signal.symbol}</span>
        <span className={`text-xs px-2 py-0.5 rounded-full border font-medium ${badge}`}>
          {signal.action}
        </span>
      </div>
      <p className="text-xs text-zinc-500 mb-2 line-clamp-2">{signal.rationale}</p>
      <div className="flex gap-3 text-xs text-zinc-400">
        <span>Confidence: <strong className="text-zinc-600">{(signal.confidence * 100).toFixed(0)}%</strong></span>
        {signal.target_price && <span>Target: <strong className="text-green-600">₹{signal.target_price}</strong></span>}
        {signal.stop_loss    && <span>SL: <strong className="text-red-600">₹{signal.stop_loss}</strong></span>}
      </div>
    </div>
  );
}


// ─── Metric Card ─────────────────────────────────────────────────────────────

function MetricCard({ label, value, sub, positive }: {
  label: string; value: string; sub?: string; positive?: boolean;
}) {
  return (
    <div className="bg-zinc-50 dark:bg-zinc-800/50 rounded-xl p-4">
      <p className="text-xs text-zinc-500 mb-1">{label}</p>
      <p className={`text-2xl font-medium ${
        positive === undefined ? "text-zinc-900 dark:text-zinc-100"
        : positive ? "text-green-600" : "text-red-500"
      }`}>{value}</p>
      {sub && <p className="text-xs text-zinc-400 mt-0.5">{sub}</p>}
    </div>
  );
}


// ─── Main Dashboard ───────────────────────────────────────────────────────────

export default function Dashboard() {
  const [symbol,    setSymbol]    = useState("RELIANCE");
  const [search,    setSearch]    = useState("RELIANCE");
  const [portfolio, setPortfolio] = useState<PortfolioSummary | null>(null);
  const [signals,   setSignals]   = useState<Signal[]>([]);
  const [ltp,       setLtp]       = useState<number | null>(null);
  const PORTFOLIO_ID = 1;

  // Fetch portfolio summary
  const fetchPortfolio = useCallback(async () => {
    const r = await fetch(`/api/portfolio/${PORTFOLIO_ID}/summary`);
    if (r.ok) setPortfolio(await r.json());
  }, []);

  // Fetch signals feed
  const fetchSignals = useCallback(async () => {
    const r = await fetch("/api/signals/feed?limit=10");
    if (r.ok) setSignals(await r.json());
  }, []);

  useEffect(() => {
    fetchPortfolio();
    fetchSignals();
    const interval = setInterval(fetchPortfolio, 10_000);
    return () => clearInterval(interval);
  }, [fetchPortfolio, fetchSignals]);

  // Live LTP via WebSocket
  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws/ticks/${symbol}`);
    ws.onmessage = (e) => {
      const tick = JSON.parse(e.data);
      setLtp(tick.close);
    };
    return () => ws.close();
  }, [symbol]);

  const handleSearch = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") setSymbol(search.toUpperCase());
  };

  const pnlPositive = (portfolio?.total_pnl ?? 0) >= 0;

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-zinc-950 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-medium text-zinc-900 dark:text-zinc-100">IndiaTrader</h1>
          <p className="text-xs text-zinc-500">Paper Trading · NSE/BSE</p>
        </div>
        <div className="flex items-center gap-3">
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            onKeyDown={handleSearch}
            placeholder="Search symbol (Enter)"
            className="text-sm border border-zinc-200 dark:border-zinc-700 rounded-lg px-3 py-2 w-48 bg-white dark:bg-zinc-900"
          />
          <div className="flex items-center gap-2 text-sm">
            <span className="text-zinc-500">{symbol}</span>
            {ltp && (
              <span className="font-medium text-zinc-900 dark:text-zinc-100">
                ₹{ltp.toLocaleString("en-IN")}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Portfolio metrics */}
      {portfolio && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
          <MetricCard
            label="Cash Balance"
            value={`₹${(portfolio.cash_balance / 1_00_000).toFixed(2)}L`}
          />
          <MetricCard
            label="Portfolio Value"
            value={`₹${(portfolio.current_value / 1_00_000).toFixed(2)}L`}
          />
          <MetricCard
            label="Unrealised P&L"
            value={`${pnlPositive ? "+" : ""}₹${Math.abs(portfolio.total_pnl).toLocaleString("en-IN")}`}
            sub={`${portfolio.total_pnl_pct >= 0 ? "+" : ""}${portfolio.total_pnl_pct.toFixed(2)}%`}
            positive={pnlPositive}
          />
          <MetricCard
            label="Positions"
            value={String(portfolio.positions.length)}
            sub="open positions"
          />
        </div>
      )}

      {/* Main grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
        {/* Chart - spans 2 cols */}
        <div className="lg:col-span-2 bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 rounded-xl p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-medium">{symbol}</h2>
            <div className="flex gap-2">
              {["1D","1W","1M","3M","1Y"].map(tf => (
                <button key={tf}
                  className="text-xs px-2 py-1 rounded-md text-zinc-500 hover:bg-zinc-100 dark:hover:bg-zinc-800">
                  {tf}
                </button>
              ))}
            </div>
          </div>
          <PriceChart symbol={symbol} />
        </div>

        {/* Order panel */}
        <OrderPanel symbol={symbol} portfolioId={PORTFOLIO_ID} />
      </div>

      {/* Bottom grid: positions + signals */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Positions table */}
        <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 rounded-xl p-4">
          <h3 className="text-sm font-medium mb-3">Open Positions</h3>
          {portfolio?.positions.length === 0 && (
            <p className="text-sm text-zinc-400 text-center py-6">No open positions</p>
          )}
          <div className="space-y-2">
            {portfolio?.positions.map((p, i) => (
              <div key={i} className="flex items-center justify-between text-sm py-2 border-b border-zinc-50 dark:border-zinc-800 last:border-0">
                <div>
                  <span className="font-medium">{p.symbol}</span>
                  <span className="text-xs text-zinc-400 ml-2">{p.quantity > 0 ? "Long" : "Short"} {Math.abs(p.quantity)}</span>
                </div>
                <div className="text-right">
                  <p className={`font-medium ${p.unrealised_pnl >= 0 ? "text-green-600" : "text-red-500"}`}>
                    {p.unrealised_pnl >= 0 ? "+" : ""}₹{p.unrealised_pnl.toLocaleString("en-IN")}
                  </p>
                  <p className="text-xs text-zinc-400">{p.unrealised_pct >= 0 ? "+" : ""}{p.unrealised_pct.toFixed(2)}%</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* AI Signals feed */}
        <div className="bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 rounded-xl p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium">AI Signals</h3>
            <button onClick={fetchSignals}
              className="text-xs text-zinc-400 hover:text-zinc-600">Refresh</button>
          </div>
          <div className="space-y-2 max-h-80 overflow-y-auto pr-1">
            {signals.length === 0 && (
              <p className="text-sm text-zinc-400 text-center py-6">No signals yet</p>
            )}
            {signals.map(s => <SignalCard key={s.id} signal={s} />)}
          </div>
        </div>
      </div>
    </div>
  );
}
