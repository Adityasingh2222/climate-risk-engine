"""
optimizer.py
------------
Mean-variance-style optimization that finds, for a range of target
portfolio returns, the allocation across the current holdings that
MINIMIZES carbon transition risk exposure (the "Green Efficient
Frontier"), subject to:
    - achieving at least a target expected return
    - weights sum to 1 (fully invested)
    - long-only (no short selling)
    - an optional per-holding maximum weight (concentration limit)

This answers a concrete portfolio-management question: "Starting from my
current holdings, how much can I reduce climate transition risk without
giving up return?"

METHODOLOGY
    minimize    w^T c
    subject to  w^T mu >= target_return
                sum(w) = 1
                0 <= w_i <= max_weight

Where:
    mu = annualized historical expected return per holding (mean daily
         return x 252 trading days) - a simple, standard estimator.
    c  = carbon risk score per holding. This project uses -value_impact_pct
         from the DCF risk model (risk_model.py / portfolio.py), so a
         higher score means more carbon-exposed.

Solved with scipy's SLSQP solver — a standard method for smooth,
constrained, small-scale nonlinear optimization problems like this one.

HONEST LIMITATION (Markowitz, 1952, and the entire mean-variance
optimization literature since): historical mean return is a noisy,
backward-looking estimator of *future* expected return. Small changes in
`mu` can swing optimized weights a lot ("estimation error maximization").
A production version would use shrinkage estimators, factor models, or
forward-looking analyst views (e.g. Black-Litterman) instead of raw
historical means — worth saying out loud if asked about this in an
interview, not something to gloss over.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def compute_expected_returns_and_cov(price_df: pd.DataFrame, tickers: list,
                                      trading_days: int = TRADING_DAYS):
    """
    Annualized expected returns (mean) and covariance matrix from
    historical daily prices, for just the given tickers (ignores any
    extra factor columns like SPY/KRBN that may also be present).
    """
    returns = price_df[tickers].pct_change().dropna()
    mu = returns.mean() * trading_days
    cov = returns.cov() * trading_days
    return mu, cov


def portfolio_return(weights, mu_vec) -> float:
    return float(np.dot(weights, mu_vec))


def portfolio_variance(weights, cov_matrix) -> float:
    return float(np.dot(weights, np.dot(cov_matrix, weights)))


def minimize_carbon_for_target_return(tickers, mu: pd.Series, carbon_scores: pd.Series,
                                       target_return: float, max_weight: float = 1.0):
    """
    Solve for the long-only portfolio weights that minimize total carbon
    risk exposure subject to achieving at least `target_return`.

    Returns (weights: pd.Series, achieved_return: float, achieved_carbon: float),
    or (None, None, None) if the solver fails or the target is infeasible
    (e.g. target_return higher than any achievable combination).
    """
    n = len(tickers)
    if n == 0:
        return None, None, None

    mu_vec = mu.reindex(tickers).fillna(0.0).values
    c_vec = carbon_scores.reindex(tickers).fillna(carbon_scores.mean() if len(carbon_scores) else 0.0).values

    # Infeasible if even the single best-return asset can't hit the target.
    if target_return > mu_vec.max() + 1e-9:
        return None, None, None

    x0 = np.ones(n) / n
    bounds = [(0.0, max_weight)] * n
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: np.dot(w, mu_vec) - target_return},
    ]

    result = minimize(
        lambda w: np.dot(w, c_vec), x0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    if not result.success:
        return None, None, None

    w = np.clip(result.x, 0, None)
    total = w.sum()
    if total <= 0:
        return None, None, None
    w = w / total  # renormalize for numerical safety

    weights = pd.Series(w, index=tickers)
    return weights, portfolio_return(w, mu_vec), float(np.dot(w, c_vec))


def build_green_efficient_frontier(tickers, mu: pd.Series, carbon_scores: pd.Series,
                                    n_points: int = 12, max_weight: float = 1.0) -> pd.DataFrame:
    """
    Sweeps target returns from the minimum to the maximum single-asset
    return and solves the min-carbon portfolio at each, returning a
    DataFrame of (target_return, achieved_return, carbon_exposure) points
    — the "Green Efficient Frontier".
    """
    mu_vec = mu.reindex(tickers).dropna().values
    if len(mu_vec) == 0:
        return pd.DataFrame()

    lo, hi = float(np.min(mu_vec)), float(np.max(mu_vec))
    if hi <= lo:
        targets = [lo]
    else:
        targets = np.linspace(lo, hi, n_points)

    rows = []
    for t in targets:
        weights, achieved_return, achieved_carbon = minimize_carbon_for_target_return(
            tickers, mu, carbon_scores, t, max_weight
        )
        if weights is None:
            continue
        rows.append({
            "target_return": t,
            "achieved_return": achieved_return,
            "carbon_exposure": achieved_carbon,
            "weights": weights,
        })
    return pd.DataFrame(rows)
