"""Root conftest: shared fixtures for all Python tests."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../container"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ml"))
