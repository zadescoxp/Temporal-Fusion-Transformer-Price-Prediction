"use client";

import { useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";

import { FeatureImportance } from "@/components/FeatureImportance";
import { Hint } from "@/components/Hint";
import { Technicals } from "@/components/Technicals";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  getTrainStatus,
  type HistoryRow,
  type HorizonDays,
  type HorizonForecast,
  type InterpretabilityResponse,
  type PredictionResponse,
  type TrainStatus,
} from "@/lib/api";
import { money, percent } from "@/lib/format";

export function PredictionPanel({
  selected,
  prediction,
  interpretation,
  latest,
  loading,
  modelMissing,
  training,
  onTrain,
  onSelectHorizon,
  onRemoteFinish,
}: {
  selected: HorizonForecast | null;
  prediction: PredictionResponse | null;
  interpretation: InterpretabilityResponse | null;
  latest: HistoryRow | null;
  loading: boolean;
  modelMissing: boolean;
  training: boolean;
  onTrain: () => void;
  onSelectHorizon: (horizon: HorizonDays) => void;
  onRemoteFinish: (ticker: string, error: string | null) => void;
}) {
  const [status, setStatus] = useState<TrainStatus | null>(null);
  const discovered = useRef(false);
  const trainingRef = useRef(training);
  const finishRef = useRef(onRemoteFinish);
  trainingRef.current = training;
  finishRef.current = onRemoteFinish;

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    async function tick() {
      let delay = trainingRef.current || discovered.current ? 1000 : 5000;
      try {
        const next = await getTrainStatus();
        if (cancelled) return;
        setStatus(next);
        if (next.running && !trainingRef.current) discovered.current = true;
        if (!next.running && discovered.current) {
          discovered.current = false;
          finishRef.current(next.ticker ?? "", next.phase === "failed" ? next.error || "Training failed." : null);
          delay = 5000;
        }
      } catch {
        delay = 5000;
      }
      if (!cancelled) timer = setTimeout(tick, delay);
    }

    void tick();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  const active = training || Boolean(status?.running);
  const horizonKey = selected ? String(selected.horizon) : "1";
  const inSample = interpretation?.evaluation?.in_sample.by_horizon[horizonKey];
  const outOfSample = interpretation?.evaluation?.out_of_sample?.by_horizon[horizonKey];
  const recent =
    interpretation?.evaluation?.out_of_sample?.recent ?? interpretation?.evaluation?.in_sample.recent ?? [];
  const roundTrip = prediction?.costs.round_trip ?? 0.003;

  return (
    <div className="space-y-8">
      {loading && !prediction && !active && (
        <div className="space-y-3">
          <Skeleton className="h-40 w-full bg-[var(--line)]" />
          <Skeleton className="h-24 w-full bg-[var(--line)]" />
        </div>
      )}
      {prediction?.smoke && (
        <p className="bg-[var(--down)] px-3 py-2 text-sm text-[var(--paper)]">
          This checkpoint is a smoke test. Train a full pass before reading the path.
        </p>
      )}

      {active && <TrainingProgress status={status} />}

      {modelMissing && !active && (
        <div className="flex items-center justify-between gap-3 border-t border-[var(--line)] py-3">
          <p className="inline-flex items-center gap-1.5 text-sm text-[var(--quiet)]">
            Not trained
            <Hint text="Fits this ticker on this machine. A full run is often 30–90 minutes, longer on CPU." />
          </p>
          <Button onClick={onTrain} disabled={training} className="h-8 bg-[var(--ink)] px-3 text-[var(--paper)] hover:opacity-90">
            {training && <Loader2 className="h-4 w-4 animate-spin" />}
            Train
          </Button>
        </div>
      )}

      {selected && (
        <section className="bg-[var(--forecast)] px-5 pb-5 pt-4 text-[var(--forecast-ink)]">
          <div className="flex items-center justify-between text-sm">
            <span>{horizonName(selected.label)}</span>
            <span className="inline-flex items-center gap-1 tabular-nums">
              {selected.date}
              <Hint text={`Edge after the ${(roundTrip * 100).toFixed(2)}% round trip is ${percent(selected.edge_after_cost)}. Long only when the median clears commission and slippage.`} />
            </span>
          </div>
          <p className="mt-8 font-display text-7xl leading-[0.85] tracking-tight">{percent(selected.simple_return.q50)}</p>
          <div className="mt-6 flex items-end justify-between text-sm">
            <span className="tabular-nums">{money(selected.price.q50)}</span>
            <span>{selected.signal === "long" ? "Long" : "Flat"}</span>
          </div>
        </section>
      )}

      {selected && (
        <div>
          <Band label="Lower" price={selected.price.q10} change={selected.simple_return.q10} tone="bg-[var(--lower)]" />
          <Band label="Median" price={selected.price.q50} change={selected.simple_return.q50} tone="bg-[var(--paper)]" />
          <Band label="Upper" price={selected.price.q90} change={selected.simple_return.q90} tone="bg-[var(--upper)]" />
        </div>
      )}

      {prediction && (
        <div className="flex flex-wrap gap-x-4 gap-y-2">
          {prediction.horizons.map((item) => {
            const activeHorizon = selected?.horizon === item.horizon;
            return (
              <button
                key={item.horizon}
                type="button"
                onClick={() => onSelectHorizon(item.horizon as HorizonDays)}
                className={`text-left text-sm tabular-nums ${activeHorizon ? "text-[var(--ink)]" : "text-[var(--quiet)] hover:text-[var(--ink)]"}`}
              >
                <span className={activeHorizon ? "border-b border-[var(--ink)]" : ""}>{item.label}</span>
                <span className="ml-2">{percent(item.simple_return.q50, 1)}</span>
              </button>
            );
          })}
        </div>
      )}

      {latest && <Technicals row={latest} />}

      {(inSample || outOfSample) && (
        <section>
          <h2 className="flex items-center gap-1.5 text-sm">
            Walk-forward
            <Hint text="After costs. Direction and coverage use every origin. Sharpe and drawdown use non-overlapping holds. In-sample also chose the checkpoint. Out-of-sample starts in 2024." />
          </h2>
          <div className="mt-3 grid grid-cols-2 gap-px bg-[var(--line)]">
            <Walk title="In sample" metrics={inSample} />
            <Walk title="Since 2024" metrics={outOfSample} />
          </div>
        </section>
      )}

      <FeatureImportance
        encoder={interpretation?.encoder_variables ?? []}
        decoder={interpretation?.decoder_variables ?? []}
        statics={interpretation?.static_variables ?? []}
        attention={interpretation?.attention ?? []}
      />

      {recent.length > 0 && (
        <section>
          <h2 className="flex items-center gap-1.5 text-sm">
            Recent sessions
            <Hint text="Realized move versus the median one-day forecast. In band means the actual landed between 0.1 and 0.9." />
          </h2>
          <ul className="mt-2">
            {recent.slice(0, 8).map((row) => (
              <li key={row.date} className="flex items-baseline justify-between border-t border-[var(--line)] py-1.5 text-sm tabular-nums">
                <span className="text-[var(--quiet)]">{row.date.slice(5)}</span>
                <span>{percent(Math.exp(row.actual_log_return) - 1)}</span>
                <span className="text-[var(--quiet)]">{percent(Math.exp(row.q50) - 1)}</span>
                <span className="w-8 text-right text-[var(--quiet)]">{row.in_band ? "in" : "out"}</span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

function TrainingProgress({ status }: { status: TrainStatus | null }) {
  const percentDone = Math.max(2, Math.min(100, Math.round((status?.progress ?? 0.02) * 100)));
  const batch = status?.batch && status.batches ? `Batch ${status.batch} of ${status.batches}` : null;
  const loss = status?.val_loss ?? status?.train_loss;
  return (
    <div>
      <div className="flex items-baseline justify-between text-sm">
        <p>{status?.message || "Starting training"}</p>
        <p className="tabular-nums">{percentDone}%</p>
      </div>
      <div className="mt-2 h-1 bg-[var(--line)]" role="progressbar" aria-valuenow={percentDone} aria-valuemin={0} aria-valuemax={100}>
        <div className="h-full bg-[var(--ink)]" style={{ width: `${percentDone}%` }} />
      </div>
      <p className="mt-2 text-sm text-[var(--quiet)]">{[batch, loss != null ? loss.toFixed(4) : null].filter(Boolean).join(" · ")}</p>
    </div>
  );
}

function horizonName(label: string) {
  const names: Record<string, string> = {
    "1d": "One day",
    "1w": "One week",
    "1m": "One month",
    "3m": "Three months",
    "6m": "Six months",
    "1y": "One year",
  };
  return names[label] ?? label;
}

function Band({ label, price, change, tone }: { label: string; price: number; change: number; tone: string }) {
  return (
    <div className={`flex items-baseline justify-between px-4 py-3 text-sm text-[var(--ink)] ${tone}`}>
      <span>{label}</span>
      <span className="tabular-nums">{money(price)}</span>
      <span className="tabular-nums">{percent(change)}</span>
    </div>
  );
}

function Walk({
  title,
  metrics,
}: {
  title: string;
  metrics?: {
    directional_accuracy: number | null;
    quantile_coverage: number | null;
    sharpe: number | null;
    max_drawdown: number | null;
  };
}) {
  const cell = (value: string) => <span className="tabular-nums">{value}</span>;
  return (
    <div className="bg-[var(--paper)] p-3">
      <p className="text-sm text-[var(--quiet)]">{title}</p>
      <p className="mt-3 font-display text-3xl leading-none">
        {metrics?.directional_accuracy == null ? "—" : percent(metrics.directional_accuracy, 0)}
      </p>
      <p className="mt-1 text-xs text-[var(--quiet)]">direction</p>
      <div className="mt-3 flex justify-between text-xs text-[var(--quiet)]">
        {cell(metrics?.quantile_coverage == null ? "—" : `cover ${percent(metrics.quantile_coverage, 0)}`)}
        {cell(metrics?.sharpe == null ? "—" : metrics.sharpe.toFixed(2))}
        {cell(metrics?.max_drawdown == null ? "—" : percent(metrics.max_drawdown, 0))}
      </div>
    </div>
  );
}
