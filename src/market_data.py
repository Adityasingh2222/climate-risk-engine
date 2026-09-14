"""
market_data.py
---------------
Fetches historical daily prices for portfolio tickers plus two factors
used in the carbon beta regression:

  - Market factor: SPY (S&P 500 ETF) as a standard market-return proxy.
  - Carbon factor: KRBN (KraneShares Global Carbon ETF, which tracks
    actual traded carbon allowance futures - EU ETS, California
    Cap-and-Trade, RGGI). This is REAL, traded carbon market data, not a
    synthetic proxy - using it is what makes the "carbon beta" analysis
    a legitimate empirical exercise rather than a toy.

If live data isn't reachable (no internet / yfinance not installed), this
module falls back to a synthetic price generator that simulates each
ticker's returns as a mix of market exposure + a *known, pre-set* carbon
exposure + noise. That lets you develop and test the regression code
offline and sanity-check that it recovers betas close to the true
simulated value (a good practice to mention in an interview: "I validated
my regression pipeline against synthetic data with known ground truth").
"""

import numpy as np
import pandas as pd

MARKET_TICKER = "SPY"
CARBON_TICKER = "KRBN"  # KraneShares Global Carbon Strategy ETF (real, traded)

# Ground-truth carbon betas used ONLY for synthetic/offline demo data.
# Roughly: high emitters have negative carbon beta (they underperform when
# carbon prices rise), low emitters are close to zero.
SYNTHETIC_TRUE_CARBON_BETA = {
    "XOM": -0.55, "CVX": -0.50, "NEE": -0.30, "LIN": -0.20, "CAT": -0.15,
    "AAPL": -0.02, "MSFT": 0.01, "JPM": -0.05, "PG": -0.03, "AMZN": -0.08,
}


def fetch_price_history(tickers, period="2y", use_live=True) -> pd.DataFrame:
    """
    Returns a DataFrame of adjusted close prices, columns = tickers
    (+ SPY and KRBN), indexed by date.
    """
    all_tickers = list(dict.fromkeys(list(tickers) + [MARKET_TICKER, CARBON_TICKER]))

    if use_live:
        try:
            import yfinance as yf
            data = yf.download(all_tickers, period=period, progress=False)["Close"]
            if isinstance(data, pd.Series):
                data = data.to_frame()
            data = data.dropna(how="all")
            if not data.empty and data.shape[1] >= 2:
                return data
        except Exception:
            pass  # fall through to synthetic

    return _generate_synthetic_prices(all_tickers, n_days=504)  # ~2 trading years


def _generate_synthetic_prices(tickers, n_days=504, seed=42) -> pd.DataFrame:
    """
    Simulate correlated daily returns:
        ticker_return = market_beta * market_return
                       + true_carbon_beta * carbon_factor_return
                       + idiosyncratic noise

    Market and carbon factor returns are themselves simulated as
    independent-ish random walks with realistic daily vol.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    n_days = len(dates)  # bdate_range can return a count that differs slightly from the request

    market_returns = rng.normal(0.0004, 0.010, n_days)          # ~ SPY-like
    carbon_returns = rng.normal(0.0002, 0.018, n_days)          # carbon ETF, more volatile

    prices = {}
    prices[MARKET_TICKER] = 400 * np.cumprod(1 + market_returns)
    prices[CARBON_TICKER] = 30 * np.cumprod(1 + carbon_returns)

    for t in tickers:
        if t in (MARKET_TICKER, CARBON_TICKER):
            continue
        true_beta = SYNTHETIC_TRUE_CARBON_BETA.get(t, -0.05)
        market_beta = rng.uniform(0.8, 1.3)
        idio = rng.normal(0, 0.012, n_days)
        r = market_beta * market_returns + true_beta * carbon_returns + idio
        start_price = rng.uniform(50, 400)
        prices[t] = start_price * np.cumprod(1 + r)

    return pd.DataFrame(prices, index=dates)
