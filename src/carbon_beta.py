"""
carbon_beta.py
--------------
Estimates each stock's empirical "carbon beta" via a two-factor
regression:

    stock_return_t = alpha + beta_market * market_return_t
                            + beta_carbon * carbon_return_t + error_t

Simplified version of the academic "carbon risk factor" approach (see
Bolton & Kacperczyk, 2021/2023, in REFERENCES.md).

REVISED METHODOLOGY (v2 — see REFERENCES.md changelog):
Two econometric issues flagged in review, now addressed:

1. Newey-West (HAC) robust standard errors. Daily financial returns
   exhibit volatility clustering and autocorrelation, which makes
   textbook OLS standard errors unreliable (they understate true
   uncertainty). We now fit with cov_type='HAC' so the reported p-values
   are robust to this.

2. Multicollinearity check (VIF). SPY (market) and KRBN (carbon) returns
   are not perfectly independent — both are influenced by broad risk
   sentiment and energy-sector moves. A Variance Inflation Factor (VIF)
   is computed for the two factors; VIF > 5 is a common rule-of-thumb
   warning sign that the individual coefficients (market vs. carbon)
   may be unstable/hard to separate, even if the combined model fits well.

REMAINING LIMITATION: a 2-year daily window is still a short sample for
a slow-moving structural relationship, and doesn't capture regime shifts
(e.g., an energy price shock). A rolling-window beta (e.g. 90-day) would
show whether a stock's carbon sensitivity is changing over time — noted
as a natural extension in REFERENCES.md / README.md.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.market_data import MARKET_TICKER, CARBON_TICKER


def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Simple daily percentage returns from a price DataFrame."""
    return price_df.pct_change().dropna(how="all")


def compute_factor_vif(price_df: pd.DataFrame) -> dict:
    """
    Variance Inflation Factor for the market and carbon factors, computed
    once (it doesn't depend on which stock you're regressing). VIF > 5 is
    a common warning threshold for problematic multicollinearity; VIF > 10
    is a strong warning.
    """
    returns = compute_returns(price_df).dropna()
    if MARKET_TICKER not in returns.columns or CARBON_TICKER not in returns.columns:
        return {"vif_market": float("nan"), "vif_carbon": float("nan")}

    X = returns[[MARKET_TICKER, CARBON_TICKER]].dropna()
    X = sm.add_constant(X)
    vif_market = variance_inflation_factor(X.values, 1)
    vif_carbon = variance_inflation_factor(X.values, 2)
    return {"vif_market": vif_market, "vif_carbon": vif_carbon}


def estimate_carbon_betas(price_df: pd.DataFrame, tickers, hac_maxlags: int = 5) -> pd.DataFrame:
    """
    Run the two-factor regression for each ticker with Newey-West (HAC)
    robust standard errors.

    Returns a DataFrame indexed by ticker with:
      alpha, beta_market, beta_carbon, p_value_carbon, r_squared, n_obs
    """
    returns = compute_returns(price_df).dropna()

    if MARKET_TICKER not in returns.columns or CARBON_TICKER not in returns.columns:
        raise ValueError("Price data must include market (SPY) and carbon (KRBN) series.")

    market_ret = returns[MARKET_TICKER]
    carbon_ret = returns[CARBON_TICKER]

    rows = []
    for t in tickers:
        if t not in returns.columns:
            continue
        y = returns[t]
        X = pd.DataFrame({"market": market_ret, "carbon": carbon_ret})
        X = sm.add_constant(X)

        aligned = pd.concat([y.rename("y"), X], axis=1).dropna()
        if len(aligned) < 30:
            continue

        model = sm.OLS(aligned["y"], aligned[["const", "market", "carbon"]]).fit(
            cov_type="HAC", cov_kwds={"maxlags": hac_maxlags}
        )

        rows.append({
            "ticker": t,
            "alpha": model.params["const"],
            "beta_market": model.params["market"],
            "beta_carbon": model.params["carbon"],
            "p_value_carbon": model.pvalues["carbon"],
            "r_squared": model.rsquared,
            "n_obs": int(model.nobs),
        })

    df = pd.DataFrame(rows).set_index("ticker")
    df["significant_at_10pct"] = df["p_value_carbon"] < 0.10
    return df.sort_values("beta_carbon")
