"""Prediction and hypothesis testing framework.

Defines testable F1 hypotheses, generates pre-race predictions,
scores them against outcomes, and tracks accuracy over time.
"""

from .registry import PredictionRegistry
from .hypotheses import (
    Hypothesis,
    PoleWinnerHypothesis,
    RetirementRiskHypothesis,
    SafetyCarHypothesis,
    UndercutHypothesis,
    WetWeatherUpsetHypothesis,
    FirstLapIncidentHypothesis,
    TireStrategyHypothesis,
)
from .scorecard import Scorecard
from .backtester import Backtester
from .race_predictor import RacePredictor

__all__ = [
    "PredictionRegistry",
    "Hypothesis",
    "PoleWinnerHypothesis",
    "RetirementRiskHypothesis",
    "SafetyCarHypothesis",
    "UndercutHypothesis",
    "WetWeatherUpsetHypothesis",
    "FirstLapIncidentHypothesis",
    "TireStrategyHypothesis",
    "Scorecard",
    "Backtester",
    "RacePredictor",
]
