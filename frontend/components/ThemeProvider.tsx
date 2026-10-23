"use client";

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { Toaster } from "sonner";

import { THEMES, themeById, themeVars, type Theme, type ThemeId } from "@/lib/themes";

const STORAGE_KEY = "forecast-theme";

type ThemeContextValue = {
  id: ThemeId;
  theme: Theme;
  setTheme: (id: ThemeId) => void;
};

const ThemeContext = createContext<ThemeContextValue | null>(null);

function isThemeId(value: string | null): value is ThemeId {
  return THEMES.some((theme) => theme.id === value);
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [id, setId] = useState<ThemeId>("paper");
  const [ready, setReady] = useState(false);
  const theme = themeById(id);

  useEffect(() => {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (isThemeId(saved)) setId(saved);
    setReady(true);
  }, []);

  useEffect(() => {
    if (!ready) return;
    const root = document.documentElement;
    root.dataset.theme = id;
    for (const [key, value] of Object.entries(themeVars(theme))) {
      root.style.setProperty(key, value);
    }
    window.localStorage.setItem(STORAGE_KEY, id);
  }, [id, ready, theme]);

  return (
    <ThemeContext.Provider value={{ id, theme, setTheme: setId }}>
      {children}
      <Toaster theme={theme.mode} />
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const value = useContext(ThemeContext);
  if (!value) throw new Error("useTheme must be used inside ThemeProvider.");
  return value;
}
