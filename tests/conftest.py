"""Root conftest: shared fixtures for all Python tests."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../container"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ml"))

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_INTEGRATION") == "1":
        return
    skip = pytest.mark.skip(reason="set RUN_INTEGRATION=1 to run integration tests")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
