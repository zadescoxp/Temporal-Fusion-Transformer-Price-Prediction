"use client";

import { useEffect, useState } from "react";

import { logoUrl } from "@/lib/stocks";

export function Logo({ ticker, className = "h-4 w-4" }: { ticker: string; className?: string }) {
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setFailed(false);
  }, [ticker]);

  if (failed) {
    return <span className={`${className} inline-block rounded-full bg-[var(--ink)]`} aria-hidden />;
  }

  return (
    <img
      src={logoUrl(ticker)}
      alt=""
      className={`${className} object-contain`}
      onError={() => setFailed(true)}
    />
  );
}
