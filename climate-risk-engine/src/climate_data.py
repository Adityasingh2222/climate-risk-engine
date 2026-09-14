"""
climate_data.py
----------------
Reference data on sector-level carbon intensity (tons of CO2-equivalent
per $1 million of revenue) and Scope 3 (supply chain) multipliers.

DYNAMIC / EDITABLE: these numbers are loaded from config/sectors.yaml at
import time. Open that YAML file, change a number, save, and re-run the
app — no Python code changes needed. If the YAML file is missing or
broken, this module falls back to the hard-coded defaults below so the
project never breaks.

METHODOLOGY NOTE: figures are illustrative sector-level averages
compiled to approximate publicly available benchmarks (EPA GHGRP, CDP
sector reports, IEA sector emissions data), simplified so the model can
run on any ticker without a paid emissions data subscription. For a
rigorous version, replace with real company-level Scope 1+2 (and Scope 3)
emissions from CDP (cdp.net) or company 10-K / sustainability reports.
"""

from src.config_loader import load_yaml

# ---- Hard-coded fallback defaults (used only if sectors.yaml is missing/broken) ----
_DEFAULT_SECTORS = {
    "Energy":                  {"intensity": 850.0, "scope3_multiplier": 1.3,  "note": "Direct fuel combustion and extraction (Scope 1 dominant)."},
    "Utilities":                {"intensity": 950.0, "scope3_multiplier": 1.1,  "note": "Fossil-fuel power generation; varies hugely by fuel mix."},
    "Basic Materials":          {"intensity": 620.0, "scope3_multiplier": 1.4,  "note": "Energy-intensive processing (smelting, cement kilns, cracking)."},
    "Industrials":               {"intensity": 210.0, "scope3_multiplier": 2.5,  "note": "Mixed - heavy manufacturing higher, services-heavy industrials lower."},
    "Consumer Cyclical":        {"intensity": 90.0,  "scope3_multiplier": 4.0,  "note": "Mostly Scope 3 (supply chain) heavy, Scope 1/2 moderate."},
    "Consumer Defensive":       {"intensity": 110.0, "scope3_multiplier": 3.0,  "note": "Agriculture and packaging-driven emissions."},
    "Real Estate":               {"intensity": 60.0,  "scope3_multiplier": 1.8,  "note": "Building energy use (heating, cooling, electricity)."},
    "Financial Services":       {"intensity": 8.0,   "scope3_multiplier": 15.0, "note": "Low direct emissions; financed emissions (Scope 3) dominate."},
    "Healthcare":                 {"intensity": 45.0,  "scope3_multiplier": 3.5,  "note": "Manufacturing and logistics driven."},
    "Technology":                 {"intensity": 15.0,  "scope3_multiplier": 8.0,  "note": "Low Scope 1/2; supply chain + device use dwarf direct operations."},
    "Communication Services":   {"intensity": 20.0,  "scope3_multiplier": 3.0,  "note": "Network infrastructure and data centers."},
    "Unknown":                    {"intensity": 150.0, "scope3_multiplier": 2.0,  "note": "No sector match found - using cross-sector average."},
}


def _load_sectors() -> dict:
    config = load_yaml("sectors.yaml")
    sectors = config.get("sectors")
    if not sectors:
        return _DEFAULT_SECTORS
    # merge: any sector missing from the YAML still falls back to the default
    merged = dict(_DEFAULT_SECTORS)
    merged.update(sectors)
    return merged


SECTORS = _load_sectors()

# Kept for backwards compatibility with any code/tests expecting these names
SECTOR_CARBON_INTENSITY = {k: v.get("intensity", 150.0) for k, v in SECTORS.items()}
SECTOR_SCOPE3_MULTIPLIER = {k: v.get("scope3_multiplier", 2.0) for k, v in SECTORS.items()}
SECTOR_NOTES = {k: v.get("note", "") for k, v in SECTORS.items()}


def get_carbon_intensity(sector: str, include_scope3: bool = False) -> float:
    """
    Return tCO2e per $M revenue for a given sector string.

    include_scope3=True applies the sector's Scope 3 multiplier, giving a
    full lifecycle estimate instead of direct-operations-only. This
    matters most for sectors like Technology and Financial Services,
    where direct emissions understate true transition exposure.
    """
    entry = SECTORS.get(sector, SECTORS["Unknown"])
    base = entry.get("intensity", 150.0)
    if include_scope3:
        return base * entry.get("scope3_multiplier", 2.0)
    return base


def get_sector_note(sector: str) -> str:
    entry = SECTORS.get(sector, SECTORS["Unknown"])
    return entry.get("note", "")


def list_sectors():
    return list(SECTORS.keys())
