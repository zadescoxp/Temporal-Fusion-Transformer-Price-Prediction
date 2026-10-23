import type { ReactNode } from "react";
import type { Metadata } from "next";
import { IBM_Plex_Sans, Newsreader } from "next/font/google";

import { ThemeProvider } from "@/components/ThemeProvider";
import "./globals.css";

const sans = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-sans",
});

const display = Newsreader({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-display",
});

export const metadata: Metadata = {
  title: "Forecast",
  description: "Local multi-horizon price desk.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className={`${sans.variable} ${display.variable} font-sans antialiased`}>
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
