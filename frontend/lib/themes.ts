export type ThemeId = "paper" | "carbon" | "newsprint" | "dusk" | "signal";

export type Theme = {
  id: ThemeId;
  name: string;
  mode: "light" | "dark";
  swatch: string;
  paper: string;
  ink: string;
  muted: string;
  line: string;
  forecast: string;
  forecastInk: string;
  lower: string;
  upper: string;
  down: string;
  chartUp: string;
  chartDown: string;
  chartBand: string;
  chartGrid: string;
  chartGuide: string;
};

export const THEMES: Theme[] = [
  {
    id: "paper",
    name: "Paper",
    mode: "light",
    swatch: "#e6f34d",
    paper: "#f6f4ef",
    ink: "#141414",
    muted: "#5c574e",
    line: "#e7e1d6",
    forecast: "#e6f34d",
    forecastInk: "#141414",
    lower: "#ff5a33",
    upper: "#d7e38a",
    down: "#ff5a33",
    chartUp: "#141414",
    chartDown: "#ff5a33",
    chartBand: "rgba(230, 243, 77, 0.45)",
    chartGrid: "rgba(20, 20, 20, 0.06)",
    chartGuide: "rgba(20, 20, 20, 0.45)",
  },
  {
    id: "carbon",
    name: "Carbon",
    mode: "dark",
    swatch: "#f4f1ea",
    paper: "#111111",
    ink: "#f4f1ea",
    muted: "#a39e94",
    line: "#2c2c2c",
    forecast: "#f4f1ea",
    forecastInk: "#111111",
    lower: "#c4492c",
    upper: "#3a4032",
    down: "#ff6a45",
    chartUp: "#f4f1ea",
    chartDown: "#ff6a45",
    chartBand: "rgba(244, 241, 234, 0.16)",
    chartGrid: "rgba(244, 241, 234, 0.08)",
    chartGuide: "rgba(244, 241, 234, 0.45)",
  },
  {
    id: "newsprint",
    name: "Newsprint",
    mode: "light",
    swatch: "#111111",
    paper: "#f3f3f1",
    ink: "#111111",
    muted: "#5e5e5a",
    line: "#d9d9d4",
    forecast: "#111111",
    forecastInk: "#f7f7f5",
    lower: "#e7b2a8",
    upper: "#e4e4e0",
    down: "#b42318",
    chartUp: "#111111",
    chartDown: "#b42318",
    chartBand: "rgba(17, 17, 17, 0.08)",
    chartGrid: "rgba(17, 17, 17, 0.06)",
    chartGuide: "rgba(17, 17, 17, 0.4)",
  },
  {
    id: "dusk",
    name: "Dusk",
    mode: "dark",
    swatch: "#e7b15a",
    paper: "#1a2230",
    ink: "#f4efe6",
    muted: "#b3ab9e",
    line: "#314056",
    forecast: "#e7b15a",
    forecastInk: "#1a2230",
    lower: "#e07a5f",
    upper: "#314056",
    down: "#e07a5f",
    chartUp: "#f4efe6",
    chartDown: "#e07a5f",
    chartBand: "rgba(231, 177, 90, 0.22)",
    chartGrid: "rgba(244, 239, 230, 0.08)",
    chartGuide: "rgba(244, 239, 230, 0.4)",
  },
  {
    id: "signal",
    name: "Signal",
    mode: "light",
    swatch: "#ff4d00",
    paper: "#fafafa",
    ink: "#0e0e0e",
    muted: "#5c5c5c",
    line: "#e6e6e6",
    forecast: "#ff4d00",
    forecastInk: "#fffaf7",
    lower: "#ffd7c7",
    upper: "#ffe8df",
    down: "#ff4d00",
    chartUp: "#0e0e0e",
    chartDown: "#ff4d00",
    chartBand: "rgba(255, 77, 0, 0.16)",
    chartGrid: "rgba(14, 14, 14, 0.06)",
    chartGuide: "rgba(14, 14, 14, 0.4)",
  },
];

export function themeById(id: ThemeId) {
  return THEMES.find((theme) => theme.id === id) ?? THEMES[0];
}

export function themeVars(theme: Theme): Record<string, string> {
  return {
    "--paper": theme.paper,
    "--ink": theme.ink,
    "--quiet": theme.muted,
    "--line": theme.line,
    "--forecast": theme.forecast,
    "--forecast-ink": theme.forecastInk,
    "--lower": theme.lower,
    "--upper": theme.upper,
    "--down": theme.down,
  };
}
