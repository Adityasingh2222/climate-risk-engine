"""
data_loader.py
---------------
Pulls the fundamental data needed for the risk model for a given ticker:
  - sector classification
  - revenue (trailing twelve months)
  - EBITDA / operating margin
  - current price and market cap
  - trailing P/E (used as a simple valuation multiple)

Tries to fetch live data via yfinance. If yfinance is unavailable (no
internet, rate-limited, or ticker not found), falls back to a small
built-in sample dataset so the rest of the pipeline can still be
demoed/tested offline.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class CompanyFundamentals:
    ticker: str
    name: str
    sector: str
    revenue: float          # in USD
    ebitda_margin: float    # as a fraction, e.g. 0.25
    market_cap: float       # in USD
    pe_ratio: float
    price: float
    source: str = "live"    # "live" or "sample"


# ---------------------------------------------------------------------
# Offline fallback data so the project runs without network access.
# Numbers are approximate, illustrative figures for demo purposes only -
# replace with live data (default behavior) whenever you have internet.
# ---------------------------------------------------------------------
SAMPLE_DATA = {
    "XOM": CompanyFundamentals("XOM", "Exxon Mobil Corp", "Energy",
                                 344_600_000_000, 0.18, 480_000_000_000, 13.5, 115.0, "sample"),
    "CVX": CompanyFundamentals("CVX", "Chevron Corp", "Energy",
                                 196_900_000_000, 0.15, 290_000_000_000, 14.8, 155.0, "sample"),
    "NEE": CompanyFundamentals("NEE", "NextEra Energy", "Utilities",
                                 28_100_000_000, 0.33, 145_000_000_000, 20.1, 70.0, "sample"),
    "LIN": CompanyFundamentals("LIN", "Linde plc", "Basic Materials",
                                 33_000_000_000, 0.28, 210_000_000_000, 34.0, 440.0, "sample"),
    "CAT": CompanyFundamentals("CAT", "Caterpillar Inc", "Industrials",
                                 67_000_000_000, 0.22, 170_000_000_000, 17.6, 340.0, "sample"),
    "AAPL": CompanyFundamentals("AAPL", "Apple Inc", "Technology",
                                 385_000_000_000, 0.31, 3_400_000_000_000, 32.0, 225.0, "sample"),
    "MSFT": CompanyFundamentals("MSFT", "Microsoft Corp", "Technology",
                                 245_000_000_000, 0.45, 3_100_000_000_000, 36.0, 415.0, "sample"),
    "JPM": CompanyFundamentals("JPM", "JPMorgan Chase & Co", "Financial Services",
                                 165_000_000_000, 0.38, 600_000_000_000, 12.0, 210.0, "sample"),
    "PG": CompanyFundamentals("PG", "Procter & Gamble", "Consumer Defensive",
                                 84_000_000_000, 0.24, 380_000_000_000, 25.0, 165.0, "sample"),
    "AMZN": CompanyFundamentals("AMZN", "Amazon.com Inc", "Consumer Cyclical",
                                 590_000_000_000, 0.11, 1_900_000_000_000, 43.0, 185.0, "sample"),
}


def fetch_fundamentals(ticker: str, use_live: bool = True) -> CompanyFundamentals:
    """
    Fetch fundamentals for a ticker. Tries yfinance first (if use_live=True
    and the package/network is available), then falls back to sample data.
    """
    ticker = ticker.upper().strip()

    if use_live:
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            info = t.info

            revenue = info.get("totalRevenue")
            # yfinance sometimes omits 'ebitda' or 'trailingPE' depending on
            # the company/exchange — fall back through related fields rather
            # than failing outright.
            ebitda = info.get("ebitda") or info.get("operatingIncome") or info.get("ebit")
            market_cap = info.get("marketCap")
            pe_ratio = info.get("trailingPE") or info.get("forwardPE")
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            sector = info.get("sector", "Unknown")
            name = info.get("shortName", ticker)

            if revenue and market_cap and price:
                ebitda_margin = (ebitda / revenue) if (ebitda and revenue) else 0.15
                return CompanyFundamentals(
                    ticker=ticker,
                    name=name,
                    sector=sector or "Unknown",
                    revenue=float(revenue),
                    ebitda_margin=float(ebitda_margin),
                    market_cap=float(market_cap),
                    pe_ratio=float(pe_ratio) if pe_ratio else 20.0,
                    price=float(price),
                    source="live",
                )
        except Exception:
            pass  # fall through to sample data

    if ticker in SAMPLE_DATA:
        return SAMPLE_DATA[ticker]

    raise ValueError(
        f"No live data available (no internet or yfinance not installed) "
        f"and '{ticker}' is not in the offline sample dataset. "
        f"Available sample tickers: {', '.join(SAMPLE_DATA.keys())}"
    )


def list_sample_tickers():
    return list(SAMPLE_DATA.keys())
