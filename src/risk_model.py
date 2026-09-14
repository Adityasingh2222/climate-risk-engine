"""
risk_model.py
-------------
Converts a carbon price scenario into an estimated earnings and
valuation impact for a single company.

REVISED METHODOLOGY (v2 — see REFERENCES.md changelog):
The original version multiplied a single year's carbon cost by the
company's trailing P/E ratio. That shortcut is mathematically equivalent
to (carbon_cost / net_income) once you expand Market Cap = Net Income x
P/E — meaning it silently ignored discounting entirely and broke down for
thin-margin or loss-making companies (division by a tiny or negative
number). This version replaces it with a standard multi-year discounted
cash flow (DCF) of the carbon cost stream, which is the textbook-correct
way to convert a future cost stream into a present value:

    PV = sum_{t=valuation_year}^{horizon_year}
             [ -carbon_cost_t * (1 - tax_rate) ] / (1 + WACC)^(t - valuation_year)

Two things this fixes:
  1. Discounting: a dollar of carbon cost in 2050 is worth much less
     today than a dollar of carbon cost in 2026 — WACC discounting
     reflects that, instead of a static single-year multiple.
  2. Tax treatment: carbon costs are a deductible operating expense, so
     the after-tax cost to shareholders is (1 - tax_rate) of the pre-tax
     cost, not the full pre-tax amount.

SIMPLIFICATIONS THAT REMAIN (be ready to name these in an interview):
  - Revenue and margins are held constant over the projection window
    (no growth, no structural change) — isolates the carbon-cost effect
    holding everything else equal. A full model would project revenue
    growth and let margins evolve.
  - WACC and tax rate are single portfolio-wide defaults (8% / 21%),
    not company-specific capital structures. Real analysis would pull
    each company's actual WACC (cost of equity + after-tax cost of debt,
    weighted by capital structure) and effective tax rate.
  - This still isn't a full enterprise-value bridge (no separate debt/
    equity treatment) — it approximates the equity value impact directly
    as a fraction of market cap, which is reasonable for relative
    comparison but not a substitute for a full DCF model with a debt
    schedule.
"""

from dataclasses import dataclass
from typing import List
from src.climate_data import get_carbon_intensity, get_sector_note
from src.data_loader import CompanyFundamentals
from src.scenarios import get_carbon_price
from src.config_loader import load_yaml

_params = load_yaml("model_params.yaml").get("valuation", {})
DEFAULT_TAX_RATE = _params.get("tax_rate", 0.21)   # approx. U.S. federal statutory corporate rate
DEFAULT_WACC = _params.get("wacc", 0.08)           # typical broad-market WACC assumption
DEFAULT_VALUATION_YEAR = _params.get("default_valuation_year", 2025)
DEFAULT_HORIZON_YEAR = _params.get("default_horizon_year", 2050)


@dataclass
class AnnualImpact:
    year: int
    carbon_price: float
    annual_emissions_tco2e: float
    carbon_cost_usd: float          # pre-tax, net of pass-through
    ebitda_impact_pct: float        # this year's operating hit, undiscounted


@dataclass
class NPVImpactResult:
    ticker: str
    name: str
    sector: str
    sector_note: str
    valuation_year: int
    horizon_year: int
    wacc: float
    tax_rate: float
    pv_carbon_cost_usd: float       # present value of the after-tax cost stream
    value_impact_pct: float         # PV as a % of current market cap
    annual_path: List[AnnualImpact]


def annual_carbon_impact(company: CompanyFundamentals, carbon_price: float,
                          pass_through_rate: float = 0.0, abatement_rate: float = 0.0,
                          include_scope3: bool = False):
    """
    Single-year emissions and operating (EBITDA) impact at a given carbon
    price — this part of the original model was fine and is unchanged.
    """
    intensity = get_carbon_intensity(company.sector, include_scope3=include_scope3)
    revenue_musd = company.revenue / 1_000_000.0

    annual_emissions = revenue_musd * intensity * (1 - abatement_rate)
    gross_cost = annual_emissions * carbon_price
    net_cost = gross_cost * (1 - pass_through_rate)

    ebitda = company.revenue * company.ebitda_margin
    ebitda_impact_pct = (net_cost / ebitda) if ebitda > 0 else float("nan")

    return annual_emissions, net_cost, ebitda_impact_pct


def npv_carbon_cost_impact(
    company: CompanyFundamentals,
    scenario_key: str,
    valuation_year: int = DEFAULT_VALUATION_YEAR,
    horizon_year: int = DEFAULT_HORIZON_YEAR,
    wacc: float = DEFAULT_WACC,
    tax_rate: float = DEFAULT_TAX_RATE,
    pass_through_rate: float = 0.0,
    abatement_rate: float = 0.0,
    include_scope3: bool = False,
) -> NPVImpactResult:
    """
    Present value (as of valuation_year) of the after-tax carbon cost
    stream from valuation_year through horizon_year, expressed as a % of
    current market cap. See module docstring for the formula and its
    simplifications.
    """
    annual_path = []
    pv = 0.0

    for year in range(valuation_year, horizon_year + 1):
        carbon_price = get_carbon_price(scenario_key, year)
        annual_emissions, net_cost, ebitda_impact_pct = annual_carbon_impact(
            company, carbon_price, pass_through_rate, abatement_rate, include_scope3
        )
        after_tax_cost = net_cost * (1 - tax_rate)
        discount_factor = (1 + wacc) ** (year - valuation_year)
        pv += after_tax_cost / discount_factor

        annual_path.append(AnnualImpact(
            year=year, carbon_price=carbon_price,
            annual_emissions_tco2e=annual_emissions,
            carbon_cost_usd=net_cost,
            ebitda_impact_pct=ebitda_impact_pct,
        ))

    value_impact_pct = (-pv / company.market_cap) if company.market_cap > 0 else float("nan")

    return NPVImpactResult(
        ticker=company.ticker, name=company.name, sector=company.sector,
        sector_note=get_sector_note(company.sector),
        valuation_year=valuation_year, horizon_year=horizon_year,
        wacc=wacc, tax_rate=tax_rate,
        pv_carbon_cost_usd=pv, value_impact_pct=value_impact_pct,
        annual_path=annual_path,
    )
