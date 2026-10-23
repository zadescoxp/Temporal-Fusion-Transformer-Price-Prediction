"use client";

import { CircleHelp } from "lucide-react";

export function Hint({ text }: { text: string }) {
  return (
    <button
      type="button"
      className="group relative inline-flex h-4 w-4 shrink-0 items-center justify-center text-current/55 hover:text-current focus-visible:text-current focus-visible:outline-none"
      aria-label={text}
    >
      <CircleHelp className="h-3.5 w-3.5" strokeWidth={1.75} />
      <span
        role="tooltip"
        className="pointer-events-none absolute bottom-[calc(100%+6px)] left-1/2 z-30 hidden w-56 -translate-x-1/2 bg-[var(--ink)] px-2.5 py-2 text-left text-[11px] font-sans font-normal normal-case leading-snug tracking-normal text-[var(--paper)] group-hover:block group-focus-visible:block"
      >
        {text}
      </span>
    </button>
  );
}
