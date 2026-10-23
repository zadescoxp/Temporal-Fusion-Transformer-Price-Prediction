"use client";

import { useEffect, useRef, useState } from "react";

import { useTheme } from "@/components/ThemeProvider";
import { THEMES } from "@/lib/themes";

export function ThemeMenu() {
  const { id, setTheme } = useTheme();
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const current = THEMES.find((theme) => theme.id === id) ?? THEMES[0];

  useEffect(() => {
    function onPointer(event: MouseEvent) {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    }
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }
    window.addEventListener("mousedown", onPointer);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onPointer);
      window.removeEventListener("keydown", onKey);
    };
  }, []);

  return (
    <div ref={root} className="relative">
      <button
        type="button"
        aria-expanded={open}
        aria-haspopup="listbox"
        onClick={() => setOpen((value) => !value)}
        className="flex items-center gap-2 text-sm text-[var(--quiet)] hover:text-[var(--ink)]"
      >
        <span className="h-2.5 w-2.5" style={{ background: current.swatch }} />
        {current.name}
      </button>
      {open && (
        <ul
          role="listbox"
          aria-label="Theme"
          className="absolute right-0 z-20 mt-2 min-w-40 border border-[var(--line)] bg-[var(--paper)] py-1"
        >
          {THEMES.map((theme) => (
            <li key={theme.id}>
              <button
                type="button"
                role="option"
                aria-selected={theme.id === id}
                onClick={() => {
                  setTheme(theme.id);
                  setOpen(false);
                }}
                className={`flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm ${
                  theme.id === id ? "text-[var(--ink)]" : "text-[var(--quiet)] hover:text-[var(--ink)]"
                }`}
              >
                <span className="h-2.5 w-2.5 shrink-0" style={{ background: theme.swatch }} />
                {theme.name}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
