"""
portfolio.py
------------
Aggregates single-company NPV carbon risk results into portfolio-level
metrics, and runs a full scenario (across years) for a portfolio.

REVISED (v2): now calls npv_carbon_cost_impact (discounted, after-tax
multi-year model) instead of the old single-year P/E-multiple shortcut.
"""

from dataclasses import dataclass
from typing import List
import pandas as pd

from src.data_loader import fetch_fundamentals
from src.risk_model import npv_carbon_cost_impact, DEFAULT_WACC, DEFAULT_TAX_RATE, DEFAULT_VALUATION_YEAR
from src.scenarios import SCENARIOS


@dataclass
class Holding:
    ticker: str
    weight: float  # portfolio weight, 0-1 (should sum to 1 across holdings)


def load_portfolio(holdings: List[Holding], use_live: bool = True) -> pd.DataFrame:
    """Fetch fundamentals for every holding and return as a DataFrame."""
    rows = []
    for h in holdings:
        try:
            f = fetch_fundamentals(h.ticker, use_live=use_live)
            rows.append({
                "ticker": f.ticker,
                "name": f.name,
                "sector": f.sector,
                "weight": h.weight,
                "revenue": f.revenue,
                "ebitda_margin": f.ebitda_margin,
                "market_cap": f.market_cap,
                "pe_ratio": f.pe_ratio,
                "price": f.price,
                "data_source": f.source,
            })
        except ValueError as e:
            print(f"[warning] skipping {h.ticker}: {e}")
    return pd.DataFrame(rows)


def run_scenario_year(
    holdings: List[Holding],
    scenario_key: str,
    horizon_year: int,
    valuation_year: int = DEFAULT_VALUATION_YEAR,
    wacc: float = DEFAULT_WACC,
    tax_rate: float = DEFAULT_TAX_RATE,
    use_live: bool = True,
    pass_through_rate: float = 0.0,
    abatement_rate: float = 0.0,
    include_scope3: bool = False,
) -> pd.DataFrame:
    """
    Run the NPV carbon risk model for every holding, discounting the
    carbon cost stream from valuation_year through horizon_year. Returns
    one row per holding, including 'weighted_value_impact_pct'.

    (Named 'run_scenario_year' for backwards compatibility with the
    dashboard's "year" slider — semantically it's now "impact if the
    scenario plays out through horizon_year", not a single-year snapshot.)
    """
    rows = []

    for h in holdings:
        try:
            company = fetch_fundamentals(h.ticker, use_live=use_live)
        except ValueError as e:
            print(f"[warning] skipping {h.ticker}: {e}")
            continue

        result = npv_carbon_cost_impact(
            company, scenario_key,
            valuation_year=valuation_year, horizon_year=horizon_year,
            wacc=wacc, tax_rate=tax_rate,
            pass_through_rate=pass_through_rate, abatement_rate=abatement_rate,
            include_scope3=include_scope3,
        )
        latest = result.annual_path[-1]

        rows.append({
            "ticker": result.ticker,
            "name": result.name,
            "sector": result.sector,
            "weight": h.weight,
            "carbon_price": latest.carbon_price,
            "annual_emissions_tco2e": latest.annual_emissions_tco2e,
            "carbon_cost_usd": latest.carbon_cost_usd,
            "ebitda_impact_pct": latest.ebitda_impact_pct,
            "pv_carbon_cost_usd": result.pv_carbon_cost_usd,
            "value_impact_pct": result.value_impact_pct,
            "weighted_value_impact_pct": result.value_impact_pct * h.weight,
        })

    return pd.DataFrame(rows)


def run_full_scenario(
    holdings: List[Holding],
    scenario_key: str,
    valuation_year: int = DEFAULT_VALUATION_YEAR,
    wacc: float = DEFAULT_WACC,
    tax_rate: float = DEFAULT_TAX_RATE,
    use_live: bool = True,
    pass_through_rate: float = 0.0,
    abatement_rate: float = 0.0,
    include_scope3: bool = False,
) -> pd.DataFrame:
    """
    Run the model across every milestone year in a scenario's carbon
    price path, treating each as a horizon year (NPV of all carbon costs
    from valuation_year up to that horizon). Returns a year-by-year
    portfolio-level impact summary — naturally increasing over time as
    more years of discounted cost are included, which is the economically
    correct shape (vs. the old model's repeated single-year snapshots).
    """
    years = sorted(SCENARIOS[scenario_key]["carbon_price_path"].keys())
    summary_rows = []

    for year in years:
        if year <= valuation_year:
            continue
        df = run_scenario_year(
            holdings, scenario_key, horizon_year=year, valuation_year=valuation_year,
            wacc=wacc, tax_rate=tax_rate, use_live=use_live,
            pass_through_rate=pass_through_rate, abatement_rate=abatement_rate,
            include_scope3=include_scope3,
        )
        if df.empty:
            continue
        summary_rows.append({
            "year": year,
            "carbon_price": df["carbon_price"].iloc[0],
            "portfolio_value_impact_pct": df["weighted_value_impact_pct"].sum(),
        })

    return pd.DataFrame(summary_rows)
