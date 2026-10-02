"""life8 test options: experiment-scale tests are marked slow and skipped by default.

Run them with LIFE8_RUN_SLOW=1 python -m pytest tests/life8 -q, or select them
explicitly with -m slow.
"""
import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: multi-seed life8 experiment; skipped unless LIFE8_RUN_SLOW=1 or -m slow")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("LIFE8_RUN_SLOW") == "1" or "slow" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="slow life8 experiment: set LIFE8_RUN_SLOW=1 or use -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
