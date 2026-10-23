const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const HORIZONS = [
  { days: 1, label: "1d" },
  { days: 5, label: "1w" },
  { days: 21, label: "1m" },
  { days: 63, label: "3m" },
  { days: 126, label: "6m" },
  { days: 252, label: "1y" },
] as const;

export type HorizonDays = (typeof HORIZONS)[number]["days"];

export type HistoryRow = {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  log_return: number;
  log_volume: number;
  sma_10: number;
  sma_20: number;
  sma_30: number;
  sma_50: number;
  sma_100: number;
  sma_200: number;
  ema_10: number;
  ema_20: number;
  ema_30: number;
  ema_50: number;
  ema_100: number;
  ema_200: number;
  ichimoku_base: number;
  vwma_20: number;
  hull_9: number;
  rsi_14: number;
  stoch_k: number;
  stoch_d: number;
  cci_20: number;
  adx_14: number;
  ao: number;
  momentum_10: number;
  macd: number;
  macd_signal: number;
  stoch_rsi_k: number;
  stoch_rsi_d: number;
  williams_r: number;
  bull_power: number;
  bear_power: number;
  ultimate_osc: number;
  atr_14: number;
  vol_5: number;
  vol_20: number;
  vol_60: number;
  ret_lag_1: number;
  ret_lag_5: number;
  ret_lag_21: number;
};

export type HistoryResponse = {
  ticker: string;
  rows: HistoryRow[];
};

export type QuantileTriple = { q10: number; q50: number; q90: number };

export type HorizonForecast = {
  horizon: number;
  label: string;
  date: string;
  log_return: QuantileTriple;
  simple_return: QuantileTriple;
  price: QuantileTriple;
  signal: "long" | "flat";
  edge_after_cost: number;
};

export type ForecastPathPoint = {
  date: string;
  q10: number;
  q50: number;
  q90: number;
  log_q10: number;
  log_q50: number;
  log_q90: number;
};

export type CostModel = {
  commission_per_side: number;
  slippage_per_side: number;
  cost_per_side: number;
  round_trip: number;
};

export type PredictionResponse = {
  ticker: string;
  as_of: string;
  last_close: number;
  smoke: boolean;
  costs: CostModel;
  horizons: HorizonForecast[];
  path: ForecastPathPoint[];
  price_band_note: string;
};

export type Importance = { name: string; weight: number };
export type AttentionPoint = { offset: number; weight: number };

export type HorizonMetrics = {
  directional_accuracy: number | null;
  quantile_coverage: number | null;
  sharpe: number | null;
  max_drawdown: number | null;
  n_origins: number;
  n_trades: number;
  strategy_return: number | null;
};

export type RecentRow = {
  date: string;
  actual_log_return: number;
  q10: number;
  q50: number;
  q90: number;
  in_band: boolean;
};

export type SplitEvaluation = {
  by_horizon: Record<string, HorizonMetrics>;
  recent: RecentRow[];
};

export type InterpretabilityResponse = {
  ticker: string;
  smoke: boolean;
  costs: CostModel | null;
  split: {
    train_end_date?: string;
    validation_start_date?: string;
    validation_end_date?: string;
    oos_start_date?: string;
  };
  encoder_variables: Importance[];
  decoder_variables: Importance[];
  static_variables: Importance[];
  attention: AttentionPoint[];
  evaluation: {
    in_sample: SplitEvaluation;
    out_of_sample: SplitEvaluation | null;
  } | null;
  error?: string;
};

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      detail = response.statusText;
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export function getHistory(ticker: string, n = 756) {
  return request<HistoryResponse>(`/api/data/${encodeURIComponent(ticker)}?n=${n}`);
}

export function getPrediction(ticker: string, horizons: number[]) {
  return request<PredictionResponse>("/api/predict", {
    method: "POST",
    body: JSON.stringify({ ticker, horizons }),
  });
}

export function getInterpretability(ticker: string) {
  return request<InterpretabilityResponse>(`/api/interpretability/${encodeURIComponent(ticker)}`);
}

export type TrainStatus = {
  running: boolean;
  ticker: string | null;
  phase: "idle" | "preparing" | "training" | "scoring" | "saving" | "done" | "failed";
  message: string;
  epoch: number | null;
  max_epochs: number | null;
  batch: number | null;
  batches: number | null;
  train_loss: number | null;
  val_loss: number | null;
  progress: number;
  error: string | null;
};

export function trainModel(ticker: string) {
  return request<unknown>("/api/train", {
    method: "POST",
    body: JSON.stringify({ ticker, max_epochs: 30, batch_size: 32, accelerator: "auto" }),
  });
}

export function getTrainStatus() {
  return request<TrainStatus>("/api/train/status");
}
