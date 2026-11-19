"""LoL game-duration forecaster: a small, leakage-aware forecasting project."""
from .model import DurationModel  # noqa: F401
from .nowcast import apply_nowcast  # noqa: F401

__version__ = "1.0.0"
