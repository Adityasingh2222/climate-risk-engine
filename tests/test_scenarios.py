"""
test_scenarios.py
------------------
Verifies the linear interpolation logic in src/scenarios.py.
"""

import pytest
from src.scenarios import get_carbon_price, SCENARIOS


class TestGetCarbonPrice:
    def test_exact_milestone_year_returns_exact_price(self):
        path = SCENARIOS["orderly"]["carbon_price_path"]
        for year, price in path.items():
            assert get_carbon_price("orderly", year) == price

    def test_midpoint_interpolation_is_linear(self):
        # orderly: 2025 -> 25, 2030 -> 45. Midpoint (2027.5) should be
        # exactly halfway: 35.
        price_2025 = get_carbon_price("orderly", 2025)
        price_2030 = get_carbon_price("orderly", 2030)
        price_mid = get_carbon_price("orderly", 2027.5)
        expected = price_2025 + (price_2030 - price_2025) * 0.5
        assert price_mid == pytest.approx(expected, rel=1e-6)

    def test_year_before_first_milestone_clamps_to_first_value(self):
        first_year = min(SCENARIOS["orderly"]["carbon_price_path"].keys())
        first_price = SCENARIOS["orderly"]["carbon_price_path"][first_year]
        assert get_carbon_price("orderly", first_year - 10) == first_price

    def test_year_after_last_milestone_clamps_to_last_value(self):
        last_year = max(SCENARIOS["orderly"]["carbon_price_path"].keys())
        last_price = SCENARIOS["orderly"]["carbon_price_path"][last_year]
        assert get_carbon_price("orderly", last_year + 10) == last_price

    def test_price_is_monotonically_nondecreasing_in_all_scenarios(self):
        """Every bundled scenario should have non-decreasing carbon
        prices over time (a reasonable sanity check for these particular
        scenario definitions, even though the interpolator itself doesn't
        enforce this)."""
        for key in SCENARIOS:
            path = SCENARIOS[key]["carbon_price_path"]
            years = sorted(path.keys())
            prices = [path[y] for y in years]
            assert all(p2 >= p1 for p1, p2 in zip(prices, prices[1:])), f"{key} prices should not decrease"

    def test_disorderly_more_volatile_shape_than_orderly(self):
        """Disorderly transition should have a smaller early price than
        orderly (delayed action) but a larger late price (sharper
        catch-up), which is the defining shape of this scenario."""
        early_orderly = get_carbon_price("orderly", 2025)
        early_disorderly = get_carbon_price("disorderly", 2025)
        late_orderly = get_carbon_price("orderly", 2050)
        late_disorderly = get_carbon_price("disorderly", 2050)
        assert early_disorderly < early_orderly
        assert late_disorderly > late_orderly
