"""
scenarios.py
------------
Carbon-price pathways inspired by the NGFS (Network for Greening the
Financial System) scenario framework.

DYNAMIC / EDITABLE: loaded from config/scenarios.yaml at import time.
Add a year, change a milestone price, or add a whole new scenario in
that YAML file — no Python code changes needed. Falls back to the
hard-coded defaults below if the config file is missing or broken.

For the official, detailed NGFS dataset (rather than this project's
simplified illustrative paths), see the NGFS Scenario Explorer:
https://data.ene.iiasa.ac.at/ngfs/
"""

from src.config_loader import load_yaml

_DEFAULT_SCENARIOS = {
    "orderly": {
        "label": "Orderly Transition",
        "description": "Climate policy tightens early and predictably. Markets have time to adjust; carbon prices rise smoothly.",
        "carbon_price_path": {2025: 25, 2030: 45, 2035: 70, 2040: 100, 2045: 130, 2050: 160},
    },
    "disorderly": {
        "label": "Disorderly Transition",
        "description": "Policy action is delayed, then arrives suddenly and forcefully to catch up on climate targets. Sharp, late carbon price shock.",
        "carbon_price_path": {2025: 10, 2030: 15, 2035: 90, 2040: 160, 2045: 200, 2050: 230},
    },
    "hothouse": {
        "label": "Hothouse World (Current Policies)",
        "description": "Little new climate policy is introduced. Carbon prices stay low, but physical climate damages (not priced here) accumulate.",
        "carbon_price_path": {2025: 8, 2030: 12, 2035: 18, 2040: 25, 2045: 32, 2050: 40},
    },
}


def _load_scenarios() -> dict:
    config = load_yaml("scenarios.yaml")
    scenarios = config.get("scenarios")
    if not scenarios:
        return _DEFAULT_SCENARIOS
    # normalize: YAML keys for carbon_price_path come in as ints already via safe_load
    merged = {}
    for key, default_val in _DEFAULT_SCENARIOS.items():
        merged[key] = scenarios.get(key, default_val)
    # include any brand-new scenarios the user added in the YAML that aren't in defaults
    for key, val in scenarios.items():
        if key not in merged:
            merged[key] = val
    return merged


SCENARIOS = _load_scenarios()


def get_carbon_price(scenario_key: str, year: int) -> float:
    """
    Linearly interpolate the carbon price ($/tCO2e) for a given scenario
    and year from the milestone data points in config/scenarios.yaml.
    """
    path = SCENARIOS[scenario_key]["carbon_price_path"]
    years = sorted(path.keys())

    if year <= years[0]:
        return path[years[0]]
    if year >= years[-1]:
        return path[years[-1]]

    for y0, y1 in zip(years, years[1:]):
        if y0 <= year <= y1:
            p0, p1 = path[y0], path[y1]
            frac = (year - y0) / (y1 - y0)
            return p0 + frac * (p1 - p0)

    return path[years[-1]]


def list_scenarios():
    return {k: v["label"] for k, v in SCENARIOS.items()}
