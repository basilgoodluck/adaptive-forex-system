import axios from "axios";

const API_BASE_URL = import.meta.env.PUBLIC_API_BASE_URL as string | undefined;
// If no backend is configured, never attempt a network call — hitting a
// relative path with no baseURL can 200 with an unrelated page instead of
// failing cleanly, which defeats the mock fallback below. Just go straight
// to mock data.
const USE_MOCK_ONLY = !API_BASE_URL;

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
});

export interface RegimeUpdate {
  time: string;
  symbol: string;
  regime_label: "TREND_CALM" | "TREND_VOLATILE" | "RANGE_CALM" | "RANGE_VOLATILE" | "BREAKOUT";
  prob_trend_calm: number;
  prob_trend_volatile: number;
  prob_range_calm: number;
  prob_range_volatile: number;
  prob_breakout: number;
  confidence: number;
  log_likelihood: number | null;
  atr_percentile: number | null;
}

export interface DcEvent {
  time: string;
  symbol: string;
  event_type: "UP" | "DOWN";
  price: number;
  extreme_price: number;
  overshoot_ratio: number | null;
  duration_bars: number | null;
  directional_imbalance: number | null;
  realized_volatility: number | null;
  adx: number | null;
}

export interface RawSignal {
  time: string;
  symbol: string;
  active_strategy: "TREND_FOLLOWING" | "MEAN_REVERSION" | "BREAKOUT";
  weighted_signal: number;
  weight_trend_following: number;
  weight_mean_reversion: number;
  weight_breakout: number;
}

export interface Bar {
  time: string;
  symbol: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Trade {
  time: string;
  symbol: string;
  side: "BUY" | "SELL";
  entry_price: number;
  exit_price: number | null;
  size: number;
  pnl: number | null;
  status: "OPEN" | "CLOSED" | "CANCELLED";
  strategy: "TREND_FOLLOWING" | "MEAN_REVERSION" | "BREAKOUT";
}

export interface EquityPoint {
  time: string;
  equity: number;
  balance: number;
  drawdown_pct: number;
}

export interface RangeQuery {
  symbol?: string;
  from?: string;
  to?: string;
  limit?: number;
}

// ---------------------------------------------------------------------
// Mock data — used as a fallback whenever the real API is unreachable
// (no base URL configured, backend down, timeout, 4xx/5xx, or a 200
// with an empty payload — see withMockFallback below).
// ---------------------------------------------------------------------

const SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY"] as const;
const REGIME_LABELS: RegimeUpdate["regime_label"][] = [
  "TREND_CALM",
  "TREND_VOLATILE",
  "RANGE_CALM",
  "RANGE_VOLATILE",
  "BREAKOUT",
];
const STRATEGIES: RawSignal["active_strategy"][] = ["TREND_FOLLOWING", "MEAN_REVERSION", "BREAKOUT"];

function seedPrice(symbol: string): number {
  if (symbol === "GBPUSD") return 1.271;
  if (symbol === "USDJPY") return 148.9;
  return 1.0842;
}

function pipSize(symbol: string): number {
  return symbol === "USDJPY" ? 0.01 : 0.0001;
}

function decimals(symbol: string): number {
  return symbol === "USDJPY" ? 3 : 5;
}

function mockRegimeProbs(label: RegimeUpdate["regime_label"]) {
  const base = {
    prob_trend_calm: 0,
    prob_trend_volatile: 0,
    prob_range_calm: 0,
    prob_range_volatile: 0,
    prob_breakout: 0,
  };
  const key = ("prob_" + label.toLowerCase()) as keyof typeof base;
  const dominant = 0.55 + Math.random() * 0.35;
  const remainder = 1 - dominant;
  const others = (Object.keys(base) as (keyof typeof base)[]).filter((k) => k !== key);
  const raw = others.map(() => Math.random());
  const rawSum = raw.reduce((a, b) => a + b, 0);
  others.forEach((k, i) => (base[k] = +((raw[i] / rawSum) * remainder).toFixed(4)));
  base[key] = +dominant.toFixed(4);
  return base;
}

function mockRegimes(params: RangeQuery = {}): RegimeUpdate[] {
  const symbols = params.symbol ? [params.symbol] : [...SYMBOLS];
  const count = params.limit ?? 96 * 90; // ~90 days of M15 bars
  const now = Date.now();
  const stepMs = 15 * 60 * 1000;
  const out: RegimeUpdate[] = [];

  for (const symbol of symbols) {
    let label = REGIME_LABELS[Math.floor(Math.random() * REGIME_LABELS.length)];
    let sinceSwitch = 0;
    for (let i = count - 1; i >= 0; i--) {
      sinceSwitch++;
      if (sinceSwitch > 6 + Math.floor(Math.random() * 10) && Math.random() < 0.3) {
        label = REGIME_LABELS[Math.floor(Math.random() * REGIME_LABELS.length)];
        sinceSwitch = 0;
      }
      out.push({
        time: new Date(now - i * stepMs).toISOString(),
        symbol,
        regime_label: label,
        ...mockRegimeProbs(label),
        confidence: +(0.5 + Math.random() * 0.5).toFixed(4),
        log_likelihood: +(-Math.random() * 200).toFixed(2),
        atr_percentile: +(Math.random() * 100).toFixed(2),
      });
    }
  }
  return out.sort((a, b) => a.time.localeCompare(b.time));
}

function mockDcEvents(params: RangeQuery = {}): DcEvent[] {
  const symbols = params.symbol ? [params.symbol] : [...SYMBOLS];
  const count = params.limit ?? 600; // ~90 days worth of DC events at typical frequency
  const now = Date.now();
  const out: DcEvent[] = [];

  for (const symbol of symbols) {
    let price = seedPrice(symbol);
    let t = now - count * 20 * 60 * 1000;
    let type: "UP" | "DOWN" = Math.random() > 0.5 ? "UP" : "DOWN";
    for (let i = 0; i < count; i++) {
      type = type === "UP" ? "DOWN" : "UP";
      const move = (0.001 + Math.random() * 0.004) * (symbol === "USDJPY" ? 100 : 1);
      const extreme = type === "UP" ? price + move : price - move;
      t += (10 + Math.random() * 30) * 60 * 1000;
      out.push({
        time: new Date(t).toISOString(),
        symbol,
        event_type: type,
        price: +price.toFixed(decimals(symbol)),
        extreme_price: +extreme.toFixed(decimals(symbol)),
        overshoot_ratio: +(Math.random() * 1.5).toFixed(3),
        duration_bars: Math.floor(3 + Math.random() * 20),
        directional_imbalance: +(Math.random() * 2 - 1).toFixed(3),
        realized_volatility: +(Math.random() * 0.02).toFixed(5),
        adx: +(10 + Math.random() * 60).toFixed(2),
      });
      price = extreme;
    }
  }
  return out.sort((a, b) => a.time.localeCompare(b.time));
}

function mockSignals(params: RangeQuery = {}): RawSignal[] {
  const symbols = params.symbol ? [params.symbol] : [...SYMBOLS];
  const count = params.limit ?? 96 * 90; // ~90 days of M15 bars
  const now = Date.now();
  const stepMs = 15 * 60 * 1000;
  const out: RawSignal[] = [];

  for (const symbol of symbols) {
    for (let i = count - 1; i >= 0; i--) {
      const w = [Math.random(), Math.random(), Math.random()];
      const sum = w[0] + w[1] + w[2];
      const weights = w.map((x) => x / sum);
      out.push({
        time: new Date(now - i * stepMs).toISOString(),
        symbol,
        active_strategy: STRATEGIES[weights.indexOf(Math.max(...weights))],
        weighted_signal: +(Math.random() * 2 - 1).toFixed(4),
        weight_trend_following: +weights[0].toFixed(4),
        weight_mean_reversion: +weights[1].toFixed(4),
        weight_breakout: +weights[2].toFixed(4),
      });
    }
  }
  return out.sort((a, b) => a.time.localeCompare(b.time));
}

// ---- NEW: OHLC bars ----
function mockBars(params: RangeQuery = {}): Bar[] {
  const symbols = params.symbol ? [params.symbol] : [...SYMBOLS];
  const count = params.limit ?? 96 * 90; // ~90 days of M15 bars
  const now = Date.now();
  const stepMs = 15 * 60 * 1000;
  const out: Bar[] = [];

  for (const symbol of symbols) {
    let close = seedPrice(symbol);
    const pip = pipSize(symbol);
    for (let i = count - 1; i >= 0; i--) {
      const open = close;
      const drift = (Math.random() - 0.5) * pip * 5;
      const wickUp = Math.random() * pip * 4;
      const wickDown = Math.random() * pip * 4;
      close = open + drift + Math.sin(i / 40) * pip * 3;
      const high = Math.max(open, close) + wickUp;
      const low = Math.min(open, close) - wickDown;
      out.push({
        time: new Date(now - i * stepMs).toISOString(),
        symbol,
        open: +open.toFixed(decimals(symbol)),
        high: +high.toFixed(decimals(symbol)),
        low: +low.toFixed(decimals(symbol)),
        close: +close.toFixed(decimals(symbol)),
        volume: Math.floor(200 + Math.random() * 1800),
      });
    }
  }
  return out.sort((a, b) => a.time.localeCompare(b.time));
}

// ---- NEW: trade / order history ----
function mockTrades(params: RangeQuery = {}): Trade[] {
  const symbols = params.symbol ? [params.symbol] : [...SYMBOLS];
  const count = params.limit ?? 500; // ~90 days of trade activity
  const now = Date.now();
  const out: Trade[] = [];

  for (const symbol of symbols) {
    let price = seedPrice(symbol);
    const pip = pipSize(symbol);
    let t = now - count * 45 * 60 * 1000;
    for (let i = 0; i < count; i++) {
      const side: Trade["side"] = Math.random() > 0.5 ? "BUY" : "SELL";
      const entry = price + (Math.random() - 0.5) * pip * 3;
      const isOpen = i === count - 1 && Math.random() < 0.3;
      const moveFavorable = Math.random() > 0.42; // slight positive edge
      const moveSize = pip * (5 + Math.random() * 40) * (moveFavorable ? 1 : -1) * (side === "BUY" ? 1 : -1);
      const exit = isOpen ? null : entry + moveSize;
      const size = +(0.1 + Math.random() * 1.9).toFixed(2);
      const pnl = exit === null ? null : +((exit - entry) * (side === "BUY" ? 1 : -1) * size * 10000).toFixed(2);

      t += (20 + Math.random() * 90) * 60 * 1000;
      out.push({
        time: new Date(t).toISOString(),
        symbol,
        side,
        entry_price: +entry.toFixed(decimals(symbol)),
        exit_price: exit === null ? null : +exit.toFixed(decimals(symbol)),
        size,
        pnl,
        status: isOpen ? "OPEN" : Math.random() < 0.04 ? "CANCELLED" : "CLOSED",
        strategy: STRATEGIES[Math.floor(Math.random() * STRATEGIES.length)],
      });
      price = exit ?? entry;
    }
  }
  return out.sort((a, b) => a.time.localeCompare(b.time));
}

// ---- NEW: account equity curve ----
function mockEquityCurve(params: RangeQuery = {}): EquityPoint[] {
  const count = params.limit ?? 90 * 48; // ~90 days at 30-minute resolution
  const now = Date.now();
  const stepMs = 30 * 60 * 1000;
  const out: EquityPoint[] = [];

  let balance = 10000;
  let equity = balance;
  let peak = equity;

  for (let i = count - 1; i >= 0; i--) {
    const pnlTick = (Math.random() - 0.47) * 35; // slight positive edge
    equity += pnlTick;
    if (Math.random() < 0.08) balance = equity; // occasional realized close
    peak = Math.max(peak, equity);
    const drawdown = peak > 0 ? ((peak - equity) / peak) * 100 : 0;

    out.push({
      time: new Date(now - i * stepMs).toISOString(),
      equity: +equity.toFixed(2),
      balance: +balance.toFixed(2),
      drawdown_pct: +drawdown.toFixed(3),
    });
  }
  return out;
}

async function withMockFallback<T extends { length: number }>(
  request: () => Promise<T>,
  fallback: () => T,
  minLength = 1
): Promise<T> {
  if (USE_MOCK_ONLY) return fallback();
  try {
    const result = await request();
    // A 200 with an empty or suspiciously thin payload (e.g. the backend
    // silently caps `limit` server-side) falls back to mock data too —
    // otherwise consumers only get a sliver of real data with no warning.
    return result && result.length >= minLength ? result : fallback();
  } catch {
    return fallback();
  }
}

export async function getRegimes(params: RangeQuery = {}): Promise<RegimeUpdate[]> {
  const minLength = params.limit ? Math.ceil(params.limit * 0.5) : 1;
  return withMockFallback(
    async () => (await api.get<RegimeUpdate[]>("/regimes", { params })).data,
    () => mockRegimes(params),
    minLength
  );
}

export async function getLatestRegimes(symbol?: string): Promise<RegimeUpdate[]> {
  return withMockFallback(
    async () => (await api.get<RegimeUpdate[]>("/regimes/latest", { params: symbol ? { symbol } : {} })).data,
    () => (symbol ? [symbol] : [...SYMBOLS]).map((s) => mockRegimes({ symbol: s, limit: 1 })[0])
  );
}

export async function getDcEvents(params: RangeQuery = {}): Promise<DcEvent[]> {
  return withMockFallback(
    async () => (await api.get<DcEvent[]>("/dc-events", { params })).data,
    () => mockDcEvents(params)
  );
}

export async function getSignals(params: RangeQuery = {}): Promise<RawSignal[]> {
  return withMockFallback(
    async () => (await api.get<RawSignal[]>("/signals", { params })).data,
    () => mockSignals(params)
  );
}

// ---- NEW: exported fetchers ----

export async function getBars(params: RangeQuery = {}): Promise<Bar[]> {
  const minLength = params.limit ? Math.ceil(params.limit * 0.5) : 1;
  return withMockFallback(
    async () => (await api.get<Bar[]>("/bars", { params })).data,
    () => mockBars(params),
    minLength
  );
}

export async function getTrades(params: RangeQuery = {}): Promise<Trade[]> {
  return withMockFallback(
    async () => (await api.get<Trade[]>("/trades", { params })).data,
    () => mockTrades(params)
  );
}

export async function getEquityCurve(params: RangeQuery = {}): Promise<EquityPoint[]> {
  return withMockFallback(
    async () => (await api.get<EquityPoint[]>("/equity", { params })).data,
    () => mockEquityCurve(params)
  );
}