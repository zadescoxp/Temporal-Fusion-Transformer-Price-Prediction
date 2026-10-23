"use client";

import type { AttentionPoint, Importance } from "@/lib/api";

import { Hint } from "@/components/Hint";

const LABELS: Record<string, string> = {
  log_return: "Log return",
  log_volume: "Volume",
  sma_10: "SMA 10",
  sma_20: "SMA 20",
  sma_30: "SMA 30",
  sma_50: "SMA 50",
  sma_100: "SMA 100",
  sma_200: "SMA 200",
  ema_10: "EMA 10",
  ema_20: "EMA 20",
  ema_30: "EMA 30",
  ema_50: "EMA 50",
  ema_100: "EMA 100",
  ema_200: "EMA 200",
  ichimoku_base: "Ichimoku base",
  vwma_20: "VWMA 20",
  hull_9: "Hull MA 9",
  rsi_14: "RSI 14",
  stoch_k: "Stochastic %K",
  stoch_d: "Stochastic %D",
  cci_20: "CCI 20",
  adx_14: "ADX 14",
  ao: "Awesome Oscillator",
  momentum_10: "Momentum 10",
  macd: "MACD 10/12",
  macd_signal: "MACD signal",
  stoch_rsi_k: "Stoch RSI %K",
  stoch_rsi_d: "Stoch RSI %D",
  williams_r: "Williams %R",
  bull_power: "Bull power",
  bear_power: "Bear power",
  ultimate_osc: "Ultimate Oscillator",
  atr_14: "ATR 14",
  vol_5: "Vol 5d",
  vol_20: "Vol 20d",
  vol_60: "Vol 60d",
  ret_lag_1: "Return lag 1",
  ret_lag_5: "Return lag 5",
  ret_lag_21: "Return lag 21",
  day_of_week: "Day of week",
  month: "Month",
  quarter: "Quarter",
  is_earnings_day: "Earnings day",
  is_dividend_day: "Dividend day",
  is_holiday: "Holiday adjacent",
  relative_time_idx: "Relative time",
  encoder_length: "Encoder length",
  log_return_center: "Return center",
  log_return_scale: "Return scale",
};

function label(name: string) {
  return LABELS[name] ?? name.replaceAll("_", " ");
}

function Bars({ items, limit = 4 }: { items: Importance[]; limit?: number }) {
  const top = [...items].sort((a, b) => b.weight - a.weight).slice(0, limit);
  const max = top[0]?.weight ?? 1;
  if (!top.length) {
    return null;
  }
  return (
    <ul className="space-y-2">
      {top.map((item) => (
        <li key={item.name}>
          <div className="mb-1 flex items-baseline justify-between gap-3 text-xs">
            <span>{label(item.name)}</span>
            <span className="font-mono text-[var(--quiet)]">{(item.weight * 100).toFixed(1)}%</span>
          </div>
          <div className="mb-1.5 h-px bg-[var(--line)]">
            <div
              className="h-px bg-[var(--ink)]"
              style={{ width: `${Math.max(8, (item.weight / max) * 100)}%` }}
            />
          </div>
        </li>
      ))}
    </ul>
  );
}

export function FeatureImportance({
  encoder,
  decoder,
  statics,
  attention,
}: {
  encoder: Importance[];
  decoder: Importance[];
  statics: Importance[];
  attention: AttentionPoint[];
}) {
  const profile = downsample(attention, 42);
  const peak = Math.max(...profile.map((point) => point.weight), 0);
  return (
    <div className="space-y-4">
      {encoder.length > 0 && (
      <div>
        <p className="mb-3 flex items-center gap-1.5 text-sm">
          What it used
          <Hint text="Share of the encoder variable-selection weights. Higher means the network leaned on that input." />
        </p>
        <Bars items={encoder} />
      </div>
      )}
      {decoder.length > 0 && (
        <div>
          <p className="mb-3 text-sm">Known ahead</p>
          <Bars items={decoder} />
        </div>
      )}
      {statics.length > 0 && (
        <div>
          <p className="mb-3 text-sm">Fixed for the series</p>
          <Bars items={statics} />
        </div>
      )}
      {profile.length > 0 && (
        <div>
          <p className="mb-3 text-sm">Attention</p>
          <div className="flex h-8 items-end gap-px" aria-hidden>
            {profile.map((point) => (
              <div
                key={point.offset}
                className="flex-1 bg-[var(--ink)]"
                style={{ height: `${Math.max(8, (point.weight / (peak || 1)) * 100)}%` }}
                title={`${point.offset}d · ${(point.weight * 100).toFixed(2)}%`}
              />
            ))}
          </div>
          <div className="mt-1 flex justify-between font-mono text-[10px] text-[var(--quiet)]">
            <span>older</span>
            <span>origin</span>
          </div>
        </div>
      )}
    </div>
  );
}

function downsample(points: AttentionPoint[], buckets: number) {
  if (points.length <= buckets) return points;
  const size = points.length / buckets;
  const grouped: AttentionPoint[] = [];
  for (let index = 0; index < buckets; index += 1) {
    const slice = points.slice(Math.floor(index * size), Math.floor((index + 1) * size));
    const weight = slice.reduce((sum, point) => sum + point.weight, 0) / (slice.length || 1);
    grouped.push({ offset: slice[0]?.offset ?? index, weight });
  }
  return grouped;
}
