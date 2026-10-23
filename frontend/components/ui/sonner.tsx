"use client";

import { Toaster as Sonner } from "sonner";

export function Toaster() {
  return (
    <Sonner
      theme="dark"
      position="top-right"
      toastOptions={{
        style: {
          background: "#111214",
          border: "1px solid rgba(255,255,255,0.08)",
          color: "#f4f4f5",
        },
      }}
    />
  );
}
