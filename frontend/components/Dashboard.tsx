"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { Loader2, RefreshCw } from "lucide-react";
import { toast } from "sonner";

import { ThemeMenu } from "@/components/ThemeMenu";
import { Hint } from "@/components/Hint";
import { Logo } from "@/components/Logo";
import { PredictionPanel } from "@/components/PredictionPanel";
import { PriceChart } from "@/components/charts/PriceChart";
import { Button } from "@/components/ui/button";
import {
  ApiError,
  getHistory,
  getInterpretability,
  getPrediction,
  HORIZONS,
  trainModel,
  type HistoryResponse,
  type HorizonDays,
  type InterpretabilityResponse,
  type PredictionResponse,
} from "@/lib/api";
import { LISTED, listedStock } from "@/lib/stocks";

export function Dashboard() {
  const [draft, setDraft] = useState("AAPL");
  const [ticker, setTicker] = useState("AAPL");
  const [horizon, setHorizon] = useState<HorizonDays>(21);
  const [history, setHistory] = useState<HistoryResponse | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [interpretation, setInterpretation] = useState<InterpretabilityResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [modelMissing, setModelMissing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (symbol: string) => {
    setLoading(true);
    setError(null);
    setPrediction(null);
    setInterpretation(null);
    setHistory((current) => (current?.ticker === symbol ? current : null));
    try {
      const nextHistory = await getHistory(symbol);
      setHistory(nextHistory);
      try {
        const [nextPrediction, nextInterpretation] = await Promise.all([
          getPrediction(
            symbol,
            HORIZONS.map((item) => item.days),
          ),
          getInterpretability(symbol),
        ]);
        setPrediction(nextPrediction);
        setInterpretation(nextInterpretation);
        setModelMissing(false);
      } catch (err) {
        setPrediction(null);
        setInterpretation(null);
        if (err instanceof ApiError && err.status === 404) {
          setModelMissing(true);
        } else if (err instanceof ApiError && err.status === 409) {
          setPrediction(null);
        } else {
          const message = err instanceof Error ? err.message : "Forecast request failed.";
          setError(message);
          toast.error(message);
        }
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Could not load prices.";
      setError(message);
      toast.error(message);
      setHistory(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load(ticker);
  }, [load, ticker]);

  async function onTrain() {
    setTraining(true);
    toast.message("Training started.");
    try {
      await trainModel(ticker);
      toast.success(`${ticker} checkpoint is ready.`);
      await load(ticker);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Training failed.";
      toast.error(message);
      setError(message);
    } finally {
      setTraining(false);
    }
  }

  function selectTicker(symbol: string) {
    const next = symbol.trim().toUpperCase();
    if (!next) return;
    setDraft(next);
    setTicker(next);
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    selectTicker(draft);
  }

  const path = useMemo(() => prediction?.path.slice(0, horizon) ?? [], [prediction, horizon]);
  const selected = prediction?.horizons.find((item) => item.horizon === horizon) ?? null;
  const session = useMemo(() => {
    const rows = history?.rows ?? [];
    if (rows.length < 2) return null;
    const last = rows[rows.length - 1];
    const previous = rows[rows.length - 2];
    return {
      close: last.close,
      change: last.close / previous.close - 1,
      date: last.date,
    };
  }, [history]);
  const quoteReady = history?.ticker === ticker;
  const lastClose = quoteReady ? (prediction?.ticker === ticker ? prediction.last_close : session?.close ?? null) : null;
  const sessionUp = (session?.change ?? 0) > 0;
  const profile = listedStock(ticker);
  const latest = quoteReady ? (history?.rows.at(-1) ?? null) : null;

  return (
    <div className="flex h-dvh flex-col bg-[var(--paper)] text-[var(--ink)]">
      <header className="flex flex-wrap items-center gap-x-6 gap-y-3 px-5 pt-4 md:px-8">
        <p className="font-display text-lg leading-none">Forecast</p>
        <nav className="order-last flex w-full min-w-0 gap-5 overflow-x-auto md:order-none md:w-auto md:flex-1" aria-label="Stocks">
          {LISTED.map((stock) => {
            const active = stock.ticker === ticker;
            return (
              <button
                key={stock.ticker}
                type="button"
                onClick={() => selectTicker(stock.ticker)}
                className={`flex shrink-0 items-center gap-2 pb-0.5 text-sm ${
                  active ? "border-b border-[var(--ink)] text-[var(--ink)]" : "text-[var(--quiet)] hover:text-[var(--ink)]"
                }`}
              >
                <Logo ticker={stock.ticker} />
                {stock.name}
              </button>
            );
          })}
        </nav>
        <div className="ml-auto flex items-center gap-4">
        <ThemeMenu />
        <form onSubmit={onSubmit} className="flex shrink-0 items-center gap-2">
          <label className="sr-only" htmlFor="ticker">
            Ticker
          </label>
          <input
            id="ticker"
            value={draft}
            onChange={(event) => setDraft(event.target.value.toUpperCase())}
            className="h-8 w-16 border-b border-[var(--ink)] bg-transparent font-sans text-sm outline-none placeholder:text-[var(--quiet)]"
            maxLength={10}
            spellCheck={false}
          />
          <Button type="submit" variant="ghost" className="h-8 px-0 text-sm hover:bg-transparent" disabled={loading || training}>
            Load
          </Button>
          <Button
            type="button"
            variant="ghost"
            className="h-8 w-8 px-0 hover:bg-transparent"
            onClick={() => void load(ticker)}
            disabled={loading || training}
            aria-label="Refresh"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
          </Button>
        </form>
        </div>
      </header>

      {error && (
        <p className="mx-5 mt-3 bg-[var(--down)] px-3 py-2 text-sm text-[var(--paper)] md:mx-8" role="alert">
          {error}
        </p>
      )}

      <main className="grid min-h-0 flex-1 gap-8 overflow-y-auto px-5 py-5 md:grid-cols-[minmax(0,1.5fr)_minmax(300px,0.72fr)] md:overflow-hidden md:px-8">
        <section className="flex min-h-[420px] flex-col md:min-h-0">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <div className="flex items-center gap-2">
                <Logo ticker={ticker} className="h-5 w-5" />
                <h1 className="font-display text-4xl leading-none tracking-tight">{profile?.name ?? ticker}</h1>
              </div>
              <p className="mt-2 max-w-md text-sm text-[var(--quiet)]">{profile?.summary ?? ticker}</p>
            </div>
            {lastClose !== null && (
              <div className="text-right">
                <p className="font-display text-4xl leading-none tracking-tight">{moneySafe(lastClose)}</p>
                {session && (
                  <p className="mt-1 text-sm tabular-nums text-[var(--quiet)]">
                    <span className={sessionUp ? "text-[var(--ink)]" : "text-[var(--down)]"}>
                      {`${session.change > 0 ? "+" : ""}${(session.change * 100).toFixed(2)}%`}
                    </span>
                    <span className="ml-2">{session.date}</span>
                  </p>
                )}
              </div>
            )}
          </div>

          <div className="mt-5 flex flex-wrap items-center justify-between gap-3 text-sm">
            <span className="inline-flex items-center gap-1 text-[var(--quiet)]">
              Path
              <Hint text="The median compounds the daily 0.5 path. The band is the 0.1–0.9 range and widens with the square root of the horizon." />
            </span>
            <div className="flex gap-4" role="tablist" aria-label="Forecast horizon">
              {HORIZONS.map((item) => (
                <button
                  key={item.days}
                  type="button"
                  role="tab"
                  aria-selected={horizon === item.days}
                  onClick={() => setHorizon(item.days)}
                  className={`text-sm tabular-nums ${
                    horizon === item.days ? "border-b border-[var(--ink)] text-[var(--ink)]" : "text-[var(--quiet)] hover:text-[var(--ink)]"
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>

          <div className="relative mt-2 min-h-0 flex-1 border-t border-[var(--line)]">
            {loading && !history ? (
              <div className="h-full min-h-[320px] animate-pulse bg-[var(--line)]" />
            ) : history && history.rows.length > 0 ? (
              <PriceChart rows={history.rows} path={path} asOf={prediction?.as_of ?? history.rows.at(-1)?.date ?? null} />
            ) : (
              <div className="flex h-full min-h-[320px] items-center justify-center text-sm text-[var(--quiet)]">No price history.</div>
            )}
          </div>
        </section>

        <aside className="min-h-0 md:overflow-y-auto md:pr-1">
          <PredictionPanel
            selected={selected}
            prediction={prediction}
            interpretation={interpretation}
            latest={latest}
            loading={loading}
            modelMissing={modelMissing}
            training={training}
            onTrain={() => void onTrain()}
            onSelectHorizon={setHorizon}
            onRemoteFinish={(symbol, failure) => {
              setTraining(false);
              if (failure) {
                toast.error(failure);
                setError(failure);
                return;
              }
              if (!symbol) return;
              toast.success(`${symbol} checkpoint is ready.`);
              setDraft(symbol);
              setTicker(symbol);
            }}
          />
        </aside>
      </main>
    </div>
  );
}

function moneySafe(value: number) {
  return value.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 });
}
