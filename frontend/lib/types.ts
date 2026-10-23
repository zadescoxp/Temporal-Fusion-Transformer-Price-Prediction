export const HORIZONS = [
  { id: 1, label: "1D" },
  { id: 5, label: "1W" },
  { id: 21, label: "1M" },
  { id: 63, label: "3M" },
  { id: 126, label: "6M" },
  { id: 252, label: "1Y" },
] as const;

export type Horizon = (typeof HORIZONS)[number]["id"];

export type Bar = {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  rsi: number;
  macd: number;
  atr: number;
};

export type HistoryResponse = {
  ticker: string;
  as_of: string;
  last_close: number;
  rows: Bar[];
};

export type HorizonForecast = {
  horizon: Horizon;
  label: string;
  date: string;
  log_return_q10: number;
  log_return_q50: number;
  log_return_q90: number;
  price_q10: number;
  price_q50: number;
  price_q90: number;
  simple_return_q50: number;
  signal: "long" | "flat";
  edge_after_cost: number;
};

export type ForecastPathPoint = {
  date: string;
  price_q10: number;
  price_q50: number;
  price_q90: number;
};

export type ForecastResponse = {
  ticker: string;
  as_of: string;
  last_close: number;
  round_trip_cost: number;
  horizons: HorizonForecast[];
  paths: ForecastPathPoint[];
};

export type Importance = { name: string; importance: number };
export type AttentionPoint = { lag: number; weight: number };

export type RecentTrade = {
  origin: string;
  predicted_simple: number;
  actual_simple: number;
  signal: "long" | "flat";
  net_simple: number;
  hit: boolean;
};

export type SliceMetrics = {
  n: number;
  n_trades: number;
  directional_accuracy: number | null;
  quantile_coverage_10_90: number | null;
  sharpe: number | null;
  max_drawdown: number | null;
  cumulative_return: number | null;
  trade_win_rate: number | null;
  recent: RecentTrade[];
};

export type EvaluationReport = {
  validation_start?: string;
  oos_start?: string | null;
  round_trip_cost?: number;
  horizons?: Record<string, { validation?: SliceMetrics; out_of_sample?: SliceMetrics }>;
};

export type InterpretationResponse = {
  ticker: string;
  encoder_variables: Importance[];
  decoder_variables: Importance[];
  static_variables: Importance[];
  attention: AttentionPoint[];
  evaluation: EvaluationReport | null;
};
