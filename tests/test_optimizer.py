"""
test_optimizer.py
------------------
Verifies src/optimizer.py — the mean-variance carbon optimizer behind
the Green Efficient Frontier.
"""

import numpy as np
import pandas as pd
import pytest

from src.optimizer import (
    minimize_carbon_for_target_return,
    build_green_efficient_frontier,
    portfolio_return,
    portfolio_variance,
)

TICKERS = ["A", "B", "C"]
MU = pd.Series({"A": 0.10, "B": 0.06, "C": 0.02}, name="mu")
# A is high-return/high-carbon, C is low-return/low-carbon, B in between
CARBON = pd.Series({"A": 0.20, "B": 0.08, "C": 0.01}, name="carbon")


class TestMinimizeCarbonForTargetReturn:
    def test_weights_sum_to_one(self):
        w, ret, carbon = minimize_carbon_for_target_return(TICKERS, MU, CARBON, target_return=0.05)
        assert w is not None
        assert w.sum() == pytest.approx(1.0, rel=1e-4)

    def test_weights_are_long_only(self):
        w, ret, carbon = minimize_carbon_for_target_return(TICKERS, MU, CARBON, target_return=0.05)
        assert (w >= -1e-6).all()

    def test_achieved_return_meets_target(self):
        w, ret, carbon = minimize_carbon_for_target_return(TICKERS, MU, CARBON, target_return=0.05)
        assert ret >= 0.05 - 1e-4

    def test_infeasible_target_returns_none(self):
        """No combination of long-only weights can exceed the single best
        asset's return (0.10) - requesting more should fail cleanly."""
        w, ret, carbon = minimize_carbon_for_target_return(TICKERS, MU, CARBON, target_return=0.50)
        assert w is None and ret is None and carbon is None

    def test_low_target_prefers_low_carbon_asset(self):
        """At a very low target return, the optimizer should be free to
        load up on C (low return, but also lowest carbon) rather than A."""
        w, ret, carbon = minimize_carbon_for_target_return(TICKERS, MU, CARBON, target_return=0.02)
        assert w["C"] > w["A"]

    def test_higher_target_return_never_decreases_min_achievable_carbon(self):
        """Core efficient-frontier property: demanding a higher return
        can only maintain or increase the minimum achievable carbon
        exposure, never decrease it (monotonicity of the frontier)."""
        _, _, carbon_low = minimize_carbon_for_target_return(TICKERS, MU, CARBON, target_return=0.03)
        _, _, carbon_high = minimize_carbon_for_target_return(TICKERS, MU, CARBON, target_return=0.08)
        assert carbon_high >= carbon_low - 1e-9

    def test_optimized_carbon_never_worse_than_equal_weight(self):
        """Sanity check against a naive baseline: for a reachable target,
        the optimizer's carbon exposure should be no worse than simple
        equal weighting."""
        target = 0.05
        w_opt, ret_opt, carbon_opt = minimize_carbon_for_target_return(TICKERS, MU, CARBON, target)
        w_equal = np.array([1 / 3, 1 / 3, 1 / 3])
        equal_return = portfolio_return(w_equal, MU.reindex(TICKERS).values)
        if equal_return >= target:
            equal_carbon = float(np.dot(w_equal, CARBON.reindex(TICKERS).values))
            assert carbon_opt <= equal_carbon + 1e-9


class TestBuildGreenEfficientFrontier:
    def test_returns_nonempty_frontier(self):
        frontier = build_green_efficient_frontier(TICKERS, MU, CARBON, n_points=8)
        assert not frontier.empty

    def test_frontier_is_monotonic_in_carbon_exposure(self):
        """As target_return increases along the frontier, carbon exposure
        should be non-decreasing - the defining shape of an efficient
        frontier trade-off curve."""
        frontier = build_green_efficient_frontier(TICKERS, MU, CARBON, n_points=10)
        frontier = frontier.sort_values("achieved_return")
        carbon_values = frontier["carbon_exposure"].values
        assert all(c2 >= c1 - 1e-6 for c1, c2 in zip(carbon_values, carbon_values[1:]))

    def test_frontier_stays_within_asset_return_bounds(self):
        frontier = build_green_efficient_frontier(TICKERS, MU, CARBON, n_points=8)
        assert frontier["achieved_return"].min() >= MU.min() - 1e-6
        assert frontier["achieved_return"].max() <= MU.max() + 1e-6


class TestPortfolioMath:
    def test_portfolio_return_is_weighted_average(self):
        w = np.array([0.5, 0.3, 0.2])
        mu_vec = np.array([0.10, 0.06, 0.02])
        expected = 0.5 * 0.10 + 0.3 * 0.06 + 0.2 * 0.02
        assert portfolio_return(w, mu_vec) == pytest.approx(expected, rel=1e-9)

    def test_portfolio_variance_matches_manual_quadratic_form(self):
        w = np.array([0.6, 0.4])
        cov = np.array([[0.04, 0.01], [0.01, 0.02]])
        expected = w @ cov @ w
        assert portfolio_variance(w, cov) == pytest.approx(expected, rel=1e-9)
