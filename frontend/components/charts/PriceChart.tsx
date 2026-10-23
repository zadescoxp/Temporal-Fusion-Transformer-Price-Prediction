"use client";

import { useEffect, useRef } from "react";
import {
  ColorType,
  CrosshairMode,
  LineStyle,
  createChart,
  type IChartApi,
  type ISeriesPrimitive,
  type SeriesAttachedParameter,
  type Time,
} from "lightweight-charts";

import type { ForecastPathPoint, HistoryRow } from "@/lib/api";
import { useTheme } from "@/components/ThemeProvider";

type BandPoint = { time: Time; lower: number; upper: number };

type BitmapScope = {
  context: CanvasRenderingContext2D;
  horizontalPixelRatio: number;
  verticalPixelRatio: number;
};

type DrawTarget = {
  useBitmapCoordinateSpace: (callback: (scope: BitmapScope) => void) => void;
};

class QuantileBand implements ISeriesPrimitive<Time> {
  private chart: IChartApi | null = null;
  private series: SeriesAttachedParameter<Time>["series"] | null = null;
  private requestUpdate: (() => void) | null = null;

  constructor(
    private points: BandPoint[],
    private today: Time | null,
    private band: string,
    private guide: string,
  ) {}

  attached(param: SeriesAttachedParameter<Time>) {
    this.chart = param.chart;
    this.series = param.series;
    this.requestUpdate = param.requestUpdate;
  }

  detached() {
    this.chart = null;
    this.series = null;
    this.requestUpdate = null;
  }

  updateAllViews() {}

  paneViews() {
    const renderer = {
      draw: (target: DrawTarget) => {
        const chart = this.chart;
        const series = this.series;
        if (!chart || !series) return;
        target.useBitmapCoordinateSpace((scope) => {
          const ctx = scope.context;
          const coordinates: { x: number; upper: number; lower: number }[] = [];
          for (const point of this.points) {
            const x = chart.timeScale().timeToCoordinate(point.time);
            const upper = series.priceToCoordinate(point.upper);
            const lower = series.priceToCoordinate(point.lower);
            if (x === null || upper === null || lower === null) continue;
            coordinates.push({ x: Number(x), upper: Number(upper), lower: Number(lower) });
          }
          if (coordinates.length < 2) return;

          const xRatio = scope.horizontalPixelRatio;
          const yRatio = scope.verticalPixelRatio;
          ctx.beginPath();
          coordinates.forEach((point, index) => {
            const x = point.x * xRatio;
            const y = Math.min(point.upper, point.lower) * yRatio;
            if (index === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          });
          for (let index = coordinates.length - 1; index >= 0; index -= 1) {
            const point = coordinates[index];
            ctx.lineTo(point.x * xRatio, Math.max(point.upper, point.lower) * yRatio);
          }
          ctx.closePath();
          ctx.fillStyle = this.band;
          ctx.fill();

          if (!this.today) return;
          const todayX = chart.timeScale().timeToCoordinate(this.today);
          if (todayX === null) return;
          ctx.beginPath();
          ctx.setLineDash([4 * xRatio, 4 * xRatio]);
          ctx.strokeStyle = this.guide;
          ctx.lineWidth = xRatio;
          ctx.moveTo(todayX * xRatio, 0);
          ctx.lineTo(todayX * xRatio, scope.context.canvas.height);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.font = `${11 * yRatio}px sans-serif`;
          ctx.fillStyle = this.guide;
          ctx.fillText("Today", todayX * xRatio + 6 * xRatio, 16 * yRatio);
        });
      },
    };
    return [
      {
        zOrder: () => "bottom" as const,
        renderer: () => renderer,
      },
    ];
  }
}

type PriceChartProps = {
  rows: HistoryRow[];
  path: ForecastPathPoint[];
  asOf: string | null;
};

export function PriceChart({ rows, path, asOf }: PriceChartProps) {
  const container = useRef<HTMLDivElement>(null);
  const { theme } = useTheme();

  useEffect(() => {
    const element = container.current;
    if (!element) return;

    const chart = createChart(element, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: theme.paper },
        textColor: theme.muted,
        fontFamily: "IBM Plex Sans, sans-serif",
      },
      grid: {
        vertLines: { visible: false },
        horzLines: { color: theme.chartGrid },
      },
      crosshair: { mode: CrosshairMode.Magnet },
      rightPriceScale: { borderColor: "transparent" },
      timeScale: { borderColor: "transparent", rightOffset: 6 },
    });

    const candles = chart.addCandlestickSeries({
      upColor: theme.chartUp,
      downColor: theme.chartDown,
      borderUpColor: theme.chartUp,
      borderDownColor: theme.chartDown,
      wickUpColor: theme.chartUp,
      wickDownColor: theme.chartDown,
      priceLineVisible: false,
    });
    const lower = chart.addLineSeries({
      color: theme.chartGuide,
      lineWidth: 1,
      lineStyle: LineStyle.Solid,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    const upper = chart.addLineSeries({
      color: theme.chartGuide,
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    const median = chart.addLineSeries({
      color: theme.ink,
      lineWidth: 2,
      priceLineVisible: true,
      lastValueVisible: true,
    });

    const candleData = rows.map((row) => ({
      time: row.date as Time,
      open: row.open,
      high: row.high,
      low: row.low,
      close: row.close,
    }));
    candles.setData(candleData);

    const anchor = rows[rows.length - 1];
    if (anchor && path.length) {
      const line = [
        { time: anchor.date as Time, value: anchor.close },
        ...path.map((point) => ({ time: point.date as Time, value: point.q50 })),
      ];
      median.setData(line);
      upper.setData([
        { time: anchor.date as Time, value: anchor.close },
        ...path.map((point) => ({ time: point.date as Time, value: point.q90 })),
      ]);
      lower.setData([
        { time: anchor.date as Time, value: anchor.close },
        ...path.map((point) => ({ time: point.date as Time, value: point.q10 })),
      ]);
      const band = new QuantileBand(
        path.map((point) => ({
          time: point.date as Time,
          lower: point.q10,
          upper: point.q90,
        })),
        (asOf ?? anchor.date) as Time,
        theme.chartBand,
        theme.chartGuide,
      );
      median.attachPrimitive(band);
    }

    const frame = requestAnimationFrame(() => {
      const from = Math.max(0, candleData.length - 252);
      chart.timeScale().setVisibleLogicalRange({
        from,
        to: candleData.length + path.length + 4,
      });
    });

    return () => {
      cancelAnimationFrame(frame);
      chart.remove();
    };
  }, [rows, path, asOf, theme]);

  return <div ref={container} className="h-full min-h-[320px] w-full" />;
}
