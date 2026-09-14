"""
config_loader.py
-----------------
Loads YAML config files from /config so the model's assumptions
(carbon prices, sector emissions intensity, WACC, tax rate, etc.) can be
edited by anyone WITHOUT touching Python code — just open the YAML file,
change a number, save, and re-run the app.

Every loader here fails safe: if a config file is missing, malformed, or
missing a key, it falls back to the hard-coded default baked into the
relevant module, and prints a warning instead of crashing. This means the
project always runs even if someone deletes or breaks a config file.
"""

import os
import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")


def load_yaml(filename: str) -> dict:
    """
    Load a YAML file from the /config directory. Returns an empty dict
    (never raises) if the file is missing or invalid, so callers can
    safely do: `load_yaml("x.yaml").get("key", hardcoded_default)`.
    """
    path = os.path.join(CONFIG_DIR, filename)
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        print(f"[config] {filename} not found in /config — using built-in defaults.")
        return {}
    except yaml.YAMLError as e:
        print(f"[config] {filename} could not be parsed ({e}) — using built-in defaults.")
        return {}
