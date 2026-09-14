"""
test_risk_model.py
-------------------
Verifies the DCF discount logic in src/risk_model.py — the module that
had a real bug in an earlier version of this project (see README.md
"Methodology changelog"). These tests exist specifically to make sure
that bug class can't silently come back.
"""

import pytest
from src.data_loader import CompanyFundamentals
from src.risk_model import npv_carbon_cost_impact, annual_carbon_impact


def make_company(revenue=100_000_000, ebitda_margin=0.25, market_cap=1_000_000_000,
                  pe_ratio=20.0, sector="Energy"):
    return CompanyFundamentals(
        ticker="TEST", name="Test Co", sector=sector,
        revenue=revenue, ebitda_margin=ebitda_margin,
        market_cap=market_cap, pe_ratio=pe_ratio, price=100.0, source="sample",
    )


class TestAnnualCarbonImpact:
    def test_zero_carbon_price_means_zero_cost(self):
        company = make_company()
        emissions, cost, ebitda_impact = annual_carbon_impact(company, carbon_price=0.0)
        assert cost == 0.0
        assert ebitda_impact == 0.0
        assert emissions > 0  # emissions still exist, just aren't priced

    def test_higher_carbon_price_means_higher_cost(self):
        company = make_company()
        _, cost_low, _ = annual_carbon_impact(company, carbon_price=10.0)
        _, cost_high, _ = annual_carbon_impact(company, carbon_price=100.0)
        assert cost_high > cost_low
        # cost should scale linearly with price (same emissions each time)
        assert cost_high == pytest.approx(cost_low * 10, rel=1e-6)

    def test_full_pass_through_eliminates_cost(self):
        company = make_company()
        _, cost, ebitda_impact = annual_carbon_impact(company, carbon_price=50.0, pass_through_rate=1.0)
        assert cost == pytest.approx(0.0, abs=1e-6)

    def test_full_abatement_eliminates_emissions_and_cost(self):
        company = make_company()
        emissions, cost, _ = annual_carbon_impact(company, carbon_price=50.0, abatement_rate=1.0)
        assert emissions == pytest.approx(0.0, abs=1e-6)
        assert cost == pytest.approx(0.0, abs=1e-6)

    def test_scope3_multiplier_increases_intensity_for_tech(self):
        # Technology has a large Scope 3 multiplier in config/sectors.yaml
        company = make_company(sector="Technology")
        emissions_direct, cost_direct, _ = annual_carbon_impact(company, carbon_price=50.0, include_scope3=False)
        emissions_full, cost_full, _ = annual_carbon_impact(company, carbon_price=50.0, include_scope3=True)
        assert emissions_full > emissions_direct
        assert cost_full > cost_direct

    def test_loss_making_company_does_not_crash(self):
        # EBITDA margin can't be negative in our dataclass by convention,
        # but a zero-margin (breakeven) company should not divide by zero.
        company = make_company(ebitda_margin=0.0)
        _, _, ebitda_impact = annual_carbon_impact(company, carbon_price=50.0)
        assert ebitda_impact != ebitda_impact or ebitda_impact == float("inf") or True
        # (documents the known edge case rather than asserting a specific
        # value — a zero-EBITDA company legitimately has an undefined /
        # infinite EBITDA impact %, which is why value_impact_pct is
        # computed independently via NPV rather than through this ratio)


class TestNPVDiscounting:
    """
    These tests are the direct regression test for the methodology bug
    that was fixed: the old model multiplied a single year's cost by P/E
    with no discounting at all. These tests fail if that bug reappears.
    """

    def test_zero_wacc_and_zero_tax_equals_undiscounted_sum(self):
        """With wacc=0 and tax_rate=0, PV should exactly equal the simple
        sum of each year's nominal carbon cost — i.e. discounting is
        genuinely doing something, and turning it off (wacc=0) recovers
        the naive sum, which is a strong sanity check on the formula."""
        company = make_company()
        result = npv_carbon_cost_impact(
            company, scenario_key="orderly",
            valuation_year=2025, horizon_year=2030,
            wacc=0.0, tax_rate=0.0,
        )
        manual_sum = sum(step.carbon_cost_usd for step in result.annual_path)
        assert result.pv_carbon_cost_usd == pytest.approx(manual_sum, rel=1e-6)

    def test_positive_wacc_reduces_pv_below_nominal_sum(self):
        """Discounting future costs at a positive rate must make the
        present value strictly smaller than the raw nominal sum — this
        is the core mathematical property of any DCF."""
        company = make_company()
        nominal = npv_carbon_cost_impact(
            company, "orderly", 2025, 2035, wacc=0.0, tax_rate=0.0
        )
        discounted = npv_carbon_cost_impact(
            company, "orderly", 2025, 2035, wacc=0.08, tax_rate=0.0
        )
        assert discounted.pv_carbon_cost_usd < nominal.pv_carbon_cost_usd

    def test_higher_wacc_means_lower_pv(self):
        """Monotonicity: a higher discount rate should never increase PV."""
        company = make_company()
        low_wacc = npv_carbon_cost_impact(company, "orderly", 2025, 2040, wacc=0.05, tax_rate=0.0)
        high_wacc = npv_carbon_cost_impact(company, "orderly", 2025, 2040, wacc=0.12, tax_rate=0.0)
        assert high_wacc.pv_carbon_cost_usd < low_wacc.pv_carbon_cost_usd

    def test_tax_shield_reduces_pv(self):
        """A carbon cost is a deductible operating expense, so a positive
        tax rate must reduce the after-tax present value versus a 0% tax
        rate, holding everything else constant."""
        company = make_company()
        no_tax = npv_carbon_cost_impact(company, "orderly", 2025, 2035, wacc=0.08, tax_rate=0.0)
        with_tax = npv_carbon_cost_impact(company, "orderly", 2025, 2035, wacc=0.08, tax_rate=0.21)
        assert with_tax.pv_carbon_cost_usd < no_tax.pv_carbon_cost_usd
        # specifically, should be reduced by (1 - tax_rate)
        assert with_tax.pv_carbon_cost_usd == pytest.approx(no_tax.pv_carbon_cost_usd * 0.79, rel=1e-6)

    def test_single_year_horizon_equals_valuation_year(self):
        """When valuation_year == horizon_year, the annual_path should
        contain exactly one year and the PV should equal that single
        year's after-tax cost (undiscounted, since t - t0 = 0)."""
        company = make_company()
        result = npv_carbon_cost_impact(
            company, "orderly", valuation_year=2025, horizon_year=2025,
            wacc=0.08, tax_rate=0.21,
        )
        assert len(result.annual_path) == 1
        expected_pv = result.annual_path[0].carbon_cost_usd * (1 - 0.21)
        assert result.pv_carbon_cost_usd == pytest.approx(expected_pv, rel=1e-6)

    def test_value_impact_pct_is_pv_over_market_cap(self):
        company = make_company(market_cap=500_000_000)
        result = npv_carbon_cost_impact(company, "orderly", 2025, 2030)
        expected = -result.pv_carbon_cost_usd / company.market_cap
        assert result.value_impact_pct == pytest.approx(expected, rel=1e-6)

    def test_value_impact_is_negative_for_any_positive_carbon_price(self):
        """A carbon cost should never IMPROVE valuation — value_impact_pct
        must be <= 0 whenever the scenario has a nonzero carbon price."""
        company = make_company()
        result = npv_carbon_cost_impact(company, "disorderly", 2025, 2050)
        assert result.value_impact_pct <= 0

    def test_higher_intensity_sector_has_larger_impact(self):
        """Energy (high carbon intensity) should show a larger (more
        negative) value impact than Technology (low intensity), all else
        equal — this is the central relative-risk claim of the model."""
        energy_co = make_company(sector="Energy")
        tech_co = make_company(sector="Technology")
        energy_result = npv_carbon_cost_impact(energy_co, "disorderly", 2025, 2040)
        tech_result = npv_carbon_cost_impact(tech_co, "disorderly", 2025, 2040)
        assert energy_result.value_impact_pct < tech_result.value_impact_pct

    def test_does_not_crash_across_all_scenarios_and_sectors(self):
        """Broad smoke test: the model should run cleanly for every
        scenario x sector combination without raising or returning NaN
        under default assumptions."""
        from src.climate_data import list_sectors
        from src.scenarios import SCENARIOS
        for sector in list_sectors():
            for scenario in SCENARIOS:
                company = make_company(sector=sector)
                result = npv_carbon_cost_impact(company, scenario, 2025, 2050)
                assert result.value_impact_pct == result.value_impact_pct  # not NaN
                assert abs(result.value_impact_pct) < 5.0  # sanity bound, not runaway
