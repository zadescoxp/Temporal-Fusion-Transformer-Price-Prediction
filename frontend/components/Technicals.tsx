"use client";

import type { HistoryRow } from "@/lib/api";
import { percent } from "@/lib/format";

import { Hint } from "@/components/Hint";

export function Technicals({ row }: { row: HistoryRow }) {
  const close = row.close;
  const histogram = (row.macd - row.macd_signal) * close;
  const rows: { label: string; value: string; hint: string }[] = [
    {
      label: "RSI 14",
      value: (row.rsi_14 * 100).toFixed(1),
      hint: "14-session relative strength, 0 to 100. Above 70 is often read as stretched, below 30 as washed out.",
    },
    {
      label: "Stochastic %K",
      value: (row.stoch_k * 100).toFixed(1),
      hint: "Where the close sits in the 14-session range, smoothed 3 and 3.",
    },
    {
      label: "MACD",
      value: signedDollars(histogram),
      hint: "10/12 MACD minus the 9-session signal, in dollars.",
    },
    {
      label: "ADX 14",
      value: (row.adx_14 * 100).toFixed(1),
      hint: "Trend strength from 0 to 100. It does not say whether the trend is up or down.",
    },
    {
      label: "Williams %R",
      value: (row.williams_r * 100).toFixed(1),
      hint: "Close versus the 14-session high, from 0 to −100.",
    },
    {
      label: "ATR 14",
      value: percent(row.atr_14, 2),
      hint: "14-session average true range, as a percent of the close.",
    },
  ];

  const averages = [
    ["SMA 20", row.sma_20],
    ["SMA 50", row.sma_50],
    ["SMA 200", row.sma_200],
    ["EMA 20", row.ema_20],
  ] as const;

  return (
    <section>
      <h2 className="flex items-center gap-1.5 text-sm">
        Technicals
        <Hint text="Latest completed session, on the usual indicator scales. Averages show how far the close sits from that line." />
      </h2>
      <dl className="mt-3">
        {rows.map((item) => (
          <div key={item.label} className="flex items-baseline justify-between gap-4 border-t border-[var(--line)] py-2 text-sm">
            <dt className="inline-flex items-center gap-1 text-[var(--quiet)]">
              {item.label}
              <Hint text={item.hint} />
            </dt>
            <dd className="tabular-nums">{item.value}</dd>
          </div>
        ))}
      </dl>
      <dl className="mt-4">
        {averages.map(([label, gap]) => {
          const versus = 1 / (1 + gap) - 1;
          return (
            <div key={label} className="flex items-baseline justify-between gap-4 border-t border-[var(--line)] py-2 text-sm">
              <dt className="text-[var(--quiet)]">{label}</dt>
              <dd className="tabular-nums">{percent(versus, 1)}</dd>
            </div>
          );
        })}
      </dl>
    </section>
  );
}

function signedDollars(value: number) {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}`;
}
