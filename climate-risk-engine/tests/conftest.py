"""
conftest.py
-----------
Makes the project root importable as `src.*` when running pytest from
anywhere (e.g. `pytest` from the repo root, or `pytest tests/` from a CI
runner), without needing an editable install.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
