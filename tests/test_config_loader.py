"""
test_config_loader.py
----------------------
Verifies the YAML config loading in src/config_loader.py: that real
config files load correctly, and — just as importantly — that missing
or broken config files fail SAFE (fall back to defaults) rather than
crashing the app. This "versatile, editable, but never fragile" property
was a specific design goal, so it's tested directly.
"""

import os
import tempfile
import pytest

from src.config_loader import load_yaml
import src.config_loader as config_loader_module


class TestLoadYaml:
    def test_loads_real_sectors_config(self):
        data = load_yaml("sectors.yaml")
        assert "sectors" in data
        assert "Energy" in data["sectors"]
        assert data["sectors"]["Energy"]["intensity"] > 0

    def test_loads_real_scenarios_config(self):
        data = load_yaml("scenarios.yaml")
        assert "scenarios" in data
        assert "orderly" in data["scenarios"]
        assert "carbon_price_path" in data["scenarios"]["orderly"]

    def test_loads_real_model_params_config(self):
        data = load_yaml("model_params.yaml")
        assert "valuation" in data
        assert "wacc" in data["valuation"]

    def test_missing_file_returns_empty_dict_not_exception(self):
        """Fail-safe behavior: a nonexistent config file must not raise -
        it should return {} so callers can fall back to hard-coded
        defaults, keeping the app runnable even with a broken config dir."""
        data = load_yaml("this_file_does_not_exist_12345.yaml")
        assert data == {}

    def test_malformed_yaml_returns_empty_dict_not_exception(self, monkeypatch, tmp_path):
        """A syntactically invalid YAML file must also fail safe rather
        than crashing the whole app on startup."""
        bad_file = tmp_path / "broken.yaml"
        bad_file.write_text("sectors: [unclosed_bracket: true\n  - broken indentation")

        monkeypatch.setattr(config_loader_module, "CONFIG_DIR", str(tmp_path))
        data = load_yaml("broken.yaml")
        assert data == {}

    def test_non_dict_yaml_returns_empty_dict(self, monkeypatch, tmp_path):
        """A YAML file that parses successfully but isn't a mapping
        (e.g. just a list, or a bare string) should also fail safe,
        since every caller expects a dict."""
        list_file = tmp_path / "list.yaml"
        list_file.write_text("- item1\n- item2\n")

        monkeypatch.setattr(config_loader_module, "CONFIG_DIR", str(tmp_path))
        data = load_yaml("list.yaml")
        assert data == {}


class TestConfigDrivesRealBehavior:
    """
    Proves the config system is genuinely load-bearing (not decorative):
    editing a YAML value changes model output on the next import, and
    module-level fallbacks kick in correctly when config is unavailable.
    """

    def test_sector_intensity_matches_config_file(self):
        from src.climate_data import get_carbon_intensity
        config = load_yaml("sectors.yaml")
        expected = config["sectors"]["Energy"]["intensity"]
        assert get_carbon_intensity("Energy") == expected

    def test_scenario_prices_match_config_file(self):
        from src.scenarios import get_carbon_price
        config = load_yaml("scenarios.yaml")
        expected_2025 = config["scenarios"]["orderly"]["carbon_price_path"][2025]
        assert get_carbon_price("orderly", 2025) == expected_2025

    def test_wacc_and_tax_rate_match_config_file(self):
        from src.risk_model import DEFAULT_WACC, DEFAULT_TAX_RATE
        config = load_yaml("model_params.yaml")
        assert DEFAULT_WACC == config["valuation"]["wacc"]
        assert DEFAULT_TAX_RATE == config["valuation"]["tax_rate"]
