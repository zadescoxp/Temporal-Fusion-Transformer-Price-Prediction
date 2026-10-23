export type ListedStock = {
  ticker: string;
  name: string;
  summary: string;
};

export const LISTED: ListedStock[] = [
  {
    ticker: "AAPL",
    name: "Apple",
    summary: "Consumer hardware and services, led by the iPhone, Mac, and the App Store.",
  },
  {
    ticker: "NVDA",
    name: "Nvidia",
    summary: "Graphics processors used for gaming, data centers, and AI training.",
  },
  {
    ticker: "MSFT",
    name: "Microsoft",
    summary: "Software and cloud, including Windows, Office, and Azure.",
  },
  {
    ticker: "GOOGL",
    name: "Alphabet",
    summary: "Google Search, YouTube, and Google Cloud.",
  },
  {
    ticker: "AMZN",
    name: "Amazon",
    summary: "Online retail, logistics, and Amazon Web Services.",
  },
  {
    ticker: "META",
    name: "Meta",
    summary: "Facebook, Instagram, WhatsApp, and Reality Labs.",
  },
];

export function listedStock(ticker: string) {
  return LISTED.find((stock) => stock.ticker === ticker);
}

/** Public ticker marks from Parqet. Swap the symbol to change the logo. */
export function logoUrl(ticker: string) {
  return `https://assets.parqet.com/logos/symbol/${encodeURIComponent(ticker)}`;
}
