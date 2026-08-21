import axios from "axios";

export const api = axios.create({
  baseURL: import.meta.env.PUBLIC_API_BASE_URL,
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

export interface RangeQuery {
  symbol?: string;
  from?: string;
  to?: string;
  limit?: number;
}

export async function getRegimes(params: RangeQuery = {}): Promise<RegimeUpdate[]> {
  const { data } = await api.get<RegimeUpdate[]>("/regimes", { params });
  return data;
}

export async function getLatestRegimes(symbol?: string): Promise<RegimeUpdate[]> {
  const { data } = await api.get<RegimeUpdate[]>("/regimes/latest", {
    params: symbol ? { symbol } : {},
  });
  return data;
}

export async function getDcEvents(params: RangeQuery = {}): Promise<DcEvent[]> {
  const { data } = await api.get<DcEvent[]>("/dc-events", { params });
  return data;
}

export async function getSignals(params: RangeQuery = {}): Promise<RawSignal[]> {
  const { data } = await api.get<RawSignal[]>("/signals", { params });
  return data;
}