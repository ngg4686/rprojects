"""Concrete, testable F1 hypotheses.

Each hypothesis defines:
- What it predicts (the question)
- How it computes a prediction from available data
- How it evaluates against the actual outcome
- What features/signals it uses

Hypotheses learn over time — each one tracks its own accuracy and
can adjust thresholds based on accumulated results.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Hypothesis(ABC):
    """Base class for a testable F1 hypothesis."""

    name: str = "base"
    description: str = ""

    @abstractmethod
    def predict(self, pre_race_data: dict) -> dict:
        """Generate a prediction given pre-race data.

        Args:
            pre_race_data: Dict containing whatever data this hypothesis needs
                (qualifying results, historical results, weather, etc.)

        Returns:
            Dict with keys: value, confidence, reasoning, type
        """

    @abstractmethod
    def evaluate(self, prediction: dict, outcome: dict) -> dict:
        """Score a prediction against the actual outcome.

        Returns:
            Dict with: correct (bool), details (str), score (float 0-1)
        """

    def required_data(self) -> list[str]:
        """List of data keys this hypothesis needs in pre_race_data."""
        return []


class PoleWinnerHypothesis(Hypothesis):
    """Will the pole sitter win the race?

    Learns the base rate per circuit — some circuits (Monaco) heavily favor
    pole; others (Interlagos) don't. Adjusts prediction based on:
    - Circuit-specific pole conversion rate
    - Pole sitter's race pace vs qualifying pace gap
    - Weather change probability
    """

    name = "pole_winner"
    description = "Predicts whether the pole position driver will win the race"

    def __init__(self, historical_results: Optional[pd.DataFrame] = None):
        self._circuit_rates: dict[str, float] = {}
        self._overall_rate: float = 0.4  # F1 average ~40%
        if historical_results is not None:
            self._fit(historical_results)

    def _fit(self, results: pd.DataFrame) -> None:
        """Learn pole conversion rates from historical data."""
        df = results.dropna(subset=["grid", "position"]).copy()
        pole_sitters = df[df["grid"] == 1]

        if len(pole_sitters) > 0:
            self._overall_rate = (pole_sitters["position"] == 1).mean()

        for circuit, group in pole_sitters.groupby("circuitId"):
            if len(group) >= 3:  # need enough data
                self._circuit_rates[circuit] = (group["position"] == 1).mean()

    def predict(self, pre_race_data: dict) -> dict:
        circuit = pre_race_data.get("circuitId", "")
        pole_driver = pre_race_data.get("pole_driver", "unknown")
        weather_risk = pre_race_data.get("weather_change_risk", 0.0)

        # Start with circuit-specific rate, fall back to overall
        base_rate = self._circuit_rates.get(circuit, self._overall_rate)

        # Adjust for weather uncertainty (rain reduces predictability)
        adjusted_rate = base_rate * (1 - 0.3 * weather_risk)

        will_win = adjusted_rate > 0.5

        reasoning_parts = [
            f"Pole sitter: {pole_driver}",
            f"Circuit pole conversion rate: {base_rate:.0%}",
        ]
        if circuit in self._circuit_rates:
            reasoning_parts.append(f"(based on {circuit} history)")
        else:
            reasoning_parts.append("(using overall F1 average)")
        if weather_risk > 0.3:
            reasoning_parts.append(f"Weather risk adjustment: -{weather_risk*30:.0f}%")

        return {
            "type": "binary",
            "value": will_win,
            "probability": adjusted_rate,
            "pole_driver": pole_driver,
            "confidence": abs(adjusted_rate - 0.5) * 2,  # higher when further from 50/50
            "reasoning": ". ".join(reasoning_parts),
        }

    def evaluate(self, prediction: dict, outcome: dict) -> dict:
        pole_won = outcome.get("winner_from_pole", False)
        predicted = prediction.get("value", False)
        prob = prediction.get("probability", 0.5)

        # Brier score component
        brier = (prob - (1.0 if pole_won else 0.0)) ** 2

        return {
            "correct": predicted == pole_won,
            "brier_score": brier,
            "details": f"Predicted {'yes' if predicted else 'no'}, actual {'yes' if pole_won else 'no'}",
            "score": 1.0 - brier,
        }

    def required_data(self) -> list[str]:
        return ["circuitId", "pole_driver"]


class RetirementRiskHypothesis(Hypothesis):
    """Which drivers are most likely to retire (DNF) from the race?

    Uses historical reliability data per team/power unit, driver DNF rates,
    and circuit characteristics (street circuits = higher incident risk).
    """

    name = "retirement_risk"
    description = "Predicts which drivers are at highest risk of not finishing"

    def __init__(self, historical_results: Optional[pd.DataFrame] = None):
        self._team_dnf_rates: dict[str, float] = {}
        self._driver_dnf_rates: dict[str, float] = {}
        self._circuit_dnf_rates: dict[str, float] = {}
        if historical_results is not None:
            self._fit(historical_results)

    def _fit(self, results: pd.DataFrame) -> None:
        df = results.copy()
        df["dnf"] = df["status"].apply(
            lambda s: s not in ("Finished", "+1 Lap", "+2 Laps", "+3 Laps")
            if pd.notna(s) else False
        )

        for team, group in df.groupby("constructorId"):
            if len(group) >= 5:
                self._team_dnf_rates[team] = group["dnf"].mean()

        for driver, group in df.groupby("driverId"):
            if len(group) >= 5:
                self._driver_dnf_rates[driver] = group["dnf"].mean()

        for circuit, group in df.groupby("circuitId"):
            if len(group) >= 10:
                self._circuit_dnf_rates[circuit] = group["dnf"].mean()

    def predict(self, pre_race_data: dict) -> dict:
        circuit = pre_race_data.get("circuitId", "")
        drivers = pre_race_data.get("drivers", [])  # list of {driverId, constructorId}

        circuit_base = self._circuit_dnf_rates.get(circuit, 0.15)

        risk_scores = []
        for entry in drivers:
            driver_id = entry.get("driverId", "")
            team_id = entry.get("constructorId", "")

            driver_rate = self._driver_dnf_rates.get(driver_id, 0.1)
            team_rate = self._team_dnf_rates.get(team_id, 0.1)

            # Combine: weight team reliability more heavily than driver
            combined = 0.4 * driver_rate + 0.4 * team_rate + 0.2 * circuit_base
            risk_scores.append({
                "driverId": driver_id,
                "constructorId": team_id,
                "dnf_probability": combined,
            })

        risk_scores.sort(key=lambda x: x["dnf_probability"], reverse=True)

        # Predict number of retirements
        expected_dnfs = sum(r["dnf_probability"] for r in risk_scores)
        high_risk = [r for r in risk_scores if r["dnf_probability"] > 0.15]

        return {
            "type": "top_n",
            "values": [r["driverId"] for r in risk_scores[:3]],
            "risk_ranking": risk_scores,
            "expected_dnfs": round(expected_dnfs, 1),
            "confidence": min(0.9, 0.3 + len(high_risk) * 0.1),
            "reasoning": (
                f"Expected ~{expected_dnfs:.1f} retirements. "
                f"Highest risk: {risk_scores[0]['driverId']} ({risk_scores[0]['dnf_probability']:.0%})"
                if risk_scores else "Insufficient data"
            ),
        }

    def evaluate(self, prediction: dict, outcome: dict) -> dict:
        predicted_top3 = set(prediction.get("values", []))
        actual_dnfs = set(outcome.get("dnf_drivers", []))
        actual_count = outcome.get("dnf_count", 0)

        hits = predicted_top3 & actual_dnfs
        expected = prediction.get("expected_dnfs", 0)
        count_error = abs(expected - actual_count)

        return {
            "correct": len(hits) > 0,
            "hits": list(hits),
            "predicted_count": expected,
            "actual_count": actual_count,
            "count_error": count_error,
            "details": f"Predicted top 3 risk: {predicted_top3}. Actual DNFs: {actual_dnfs}. Hits: {hits}",
            "score": len(hits) / max(len(predicted_top3), 1),
        }

    def required_data(self) -> list[str]:
        return ["circuitId", "drivers"]


class SafetyCarHypothesis(Hypothesis):
    """Will there be a safety car deployment?

    Based on circuit characteristics, historical SC rates, weather,
    and race start incident probability.
    """

    name = "safety_car"
    description = "Predicts likelihood of safety car deployment during the race"

    def __init__(self, historical_results: Optional[pd.DataFrame] = None):
        self._circuit_sc_rate: dict[str, float] = {}
        self._overall_rate: float = 0.65  # ~65% of modern races have SC
        # Would need race control data to properly fit this

    def predict(self, pre_race_data: dict) -> dict:
        circuit = pre_race_data.get("circuitId", "")
        is_street_circuit = pre_race_data.get("is_street_circuit", False)
        weather_risk = pre_race_data.get("weather_change_risk", 0.0)
        num_starters = pre_race_data.get("num_starters", 20)

        base_rate = self._circuit_sc_rate.get(circuit, self._overall_rate)

        # Street circuits have higher SC probability
        if is_street_circuit:
            base_rate = min(1.0, base_rate * 1.3)

        # Wet conditions dramatically increase SC probability
        adjusted = base_rate + 0.2 * weather_risk

        return {
            "type": "probability",
            "value": adjusted > 0.5,
            "probability": min(1.0, adjusted),
            "confidence": abs(adjusted - 0.5) * 2,
            "reasoning": (
                f"Circuit base SC rate: {base_rate:.0%}"
                + (f", street circuit bonus" if is_street_circuit else "")
                + (f", weather risk +{weather_risk*20:.0f}%" if weather_risk > 0.1 else "")
            ),
        }

    def evaluate(self, prediction: dict, outcome: dict) -> dict:
        had_sc = outcome.get("safety_car", False)
        predicted_prob = prediction.get("probability", 0.5)
        predicted_binary = prediction.get("value", False)

        brier = (predicted_prob - (1.0 if had_sc else 0.0)) ** 2

        return {
            "correct": predicted_binary == had_sc,
            "brier_score": brier,
            "score": 1.0 - brier,
            "details": f"Predicted {predicted_prob:.0%} chance, {'happened' if had_sc else 'no SC'}",
        }

    def required_data(self) -> list[str]:
        return ["circuitId"]


class UndercutHypothesis(Hypothesis):
    """Will the undercut be effective at this circuit?

    Depends on pit lane time loss, tire warm-up characteristics,
    and how much dirty air affects lap times (i.e., can you gain
    enough on fresh tires to overcome the delta?).
    """

    name = "undercut_effective"
    description = "Predicts whether undercutting will gain positions in this race"

    def __init__(self):
        # Circuits where undercut historically works well
        self._undercut_friendly: dict[str, float] = {}

    def predict(self, pre_race_data: dict) -> dict:
        circuit = pre_race_data.get("circuitId", "")
        pit_loss = pre_race_data.get("pit_time_loss", 22.0)  # seconds
        deg_rate = pre_race_data.get("expected_degradation", 0.05)  # sec/lap

        # Undercut works when: fresh tire advantage > time lost in pit delta
        # Rough model: undercut window = deg_rate * tire_age_at_stop / pit_loss_delta
        undercut_score = self._undercut_friendly.get(circuit, 0.5)

        # High degradation = undercut more effective
        if deg_rate > 0.08:
            undercut_score = min(1.0, undercut_score + 0.2)

        # Very long pit lanes reduce undercut effectiveness
        if pit_loss > 25:
            undercut_score = max(0.0, undercut_score - 0.15)

        return {
            "type": "probability",
            "value": undercut_score > 0.5,
            "probability": undercut_score,
            "confidence": abs(undercut_score - 0.5) * 2,
            "reasoning": (
                f"Undercut score: {undercut_score:.2f}. "
                f"Pit loss: {pit_loss:.1f}s, degradation: {deg_rate:.3f} s/lap"
            ),
        }

    def evaluate(self, prediction: dict, outcome: dict) -> dict:
        undercut_worked = outcome.get("undercut_gained_position", False)
        predicted = prediction.get("value", False)

        return {
            "correct": predicted == undercut_worked,
            "details": f"Predicted {'yes' if predicted else 'no'}, undercut {'worked' if undercut_worked else 'failed'}",
            "score": 1.0 if predicted == undercut_worked else 0.0,
        }

    def required_data(self) -> list[str]:
        return ["circuitId"]


class WetWeatherUpsetHypothesis(Hypothesis):
    """Will rain cause a significant upset in the results?

    An 'upset' = a driver finishing 10+ places higher than their grid
    position, or a non-top-5 qualifier winning.
    """

    name = "wet_weather_upset"
    description = "Predicts whether rain will cause a major result upset"

    def predict(self, pre_race_data: dict) -> dict:
        rain_probability = pre_race_data.get("rain_probability", 0.0)
        session_is_wet = pre_race_data.get("session_wet", False)

        if session_is_wet or rain_probability > 0.6:
            upset_prob = 0.35 + 0.3 * rain_probability
        else:
            upset_prob = 0.05  # upsets happen rarely in dry races too

        return {
            "type": "probability",
            "value": upset_prob > 0.3,
            "probability": min(1.0, upset_prob),
            "confidence": abs(upset_prob - 0.5) * 2,
            "reasoning": (
                f"Rain probability: {rain_probability:.0%}. "
                f"Upset probability: {upset_prob:.0%}"
            ),
        }

    def evaluate(self, prediction: dict, outcome: dict) -> dict:
        upset_occurred = outcome.get("upset_occurred", False)
        predicted = prediction.get("value", False)
        prob = prediction.get("probability", 0.5)

        brier = (prob - (1.0 if upset_occurred else 0.0)) ** 2

        return {
            "correct": predicted == upset_occurred,
            "brier_score": brier,
            "score": 1.0 - brier,
            "details": (
                f"Predicted {prob:.0%} upset chance. "
                f"{'Upset occurred' if upset_occurred else 'No upset'}: "
                + outcome.get("upset_details", "")
            ),
        }


class FirstLapIncidentHypothesis(Hypothesis):
    """Will there be a significant first-lap incident?

    Based on circuit Turn 1 characteristics, grid spread,
    and historical first-lap incident data.
    """

    name = "first_lap_incident"
    description = "Predicts likelihood of a first-lap incident affecting results"

    def __init__(self):
        # Circuits known for first-lap chaos
        self._high_risk_circuits = {
            "spa", "monza", "jeddah", "baku", "monaco",
            "albert_park", "silverstone",
        }

    def predict(self, pre_race_data: dict) -> dict:
        circuit = pre_race_data.get("circuitId", "")
        quali_spread = pre_race_data.get("quali_spread_seconds", 2.0)

        base_rate = 0.25  # ~25% of races have first-lap incidents

        if circuit in self._high_risk_circuits:
            base_rate += 0.15

        # Tight qualifying spread = closer grid = more first lap contact risk
        if quali_spread < 1.5:
            base_rate += 0.1

        return {
            "type": "probability",
            "value": base_rate > 0.35,
            "probability": min(1.0, base_rate),
            "confidence": abs(base_rate - 0.5) * 2,
            "reasoning": (
                f"Base rate: {base_rate:.0%}. "
                f"Circuit: {circuit}"
                + (f" (high-risk T1)" if circuit in self._high_risk_circuits else "")
            ),
        }

    def evaluate(self, prediction: dict, outcome: dict) -> dict:
        incident = outcome.get("first_lap_incident", False)
        predicted = prediction.get("value", False)

        return {
            "correct": predicted == incident,
            "score": 1.0 if predicted == incident else 0.0,
            "details": f"Predicted {'yes' if predicted else 'no'}, {'incident' if incident else 'clean start'}",
        }


class TireStrategyHypothesis(Hypothesis):
    """What will the optimal tire strategy be?

    Predicts 1-stop vs 2-stop as optimal, and which compounds.
    Based on circuit degradation history, temperature, and race length.
    """

    name = "tire_strategy"
    description = "Predicts the optimal pit stop strategy (1-stop vs 2-stop, compounds)"

    def __init__(self):
        self._circuit_strategies: dict[str, dict] = {}

    def predict(self, pre_race_data: dict) -> dict:
        circuit = pre_race_data.get("circuitId", "")
        track_temp = pre_race_data.get("track_temp", 35.0)
        race_laps = pre_race_data.get("race_laps", 55)
        compounds_available = pre_race_data.get("compounds", ["SOFT", "MEDIUM", "HARD"])

        # Simple heuristic — can be replaced with learned model
        # High temp + many laps = likely 2-stop
        two_stop_score = 0.3
        if track_temp > 40:
            two_stop_score += 0.2
        if race_laps > 60:
            two_stop_score += 0.15

        historical = self._circuit_strategies.get(circuit, {})
        if historical.get("typical_stops", 1) == 2:
            two_stop_score += 0.25

        is_two_stop = two_stop_score > 0.5

        if is_two_stop:
            strategy = f"2-stop: {compounds_available[0]}-{compounds_available[1]}-{compounds_available[1]}"
        else:
            strategy = f"1-stop: {compounds_available[1]}-{compounds_available[2]}"

        return {
            "type": "categorical",
            "value": 2 if is_two_stop else 1,
            "strategy_description": strategy,
            "two_stop_probability": two_stop_score,
            "confidence": abs(two_stop_score - 0.5) * 2,
            "reasoning": (
                f"Predicted {'2-stop' if is_two_stop else '1-stop'}. "
                f"Score: {two_stop_score:.2f}. "
                f"Track temp: {track_temp}°C, race length: {race_laps} laps"
            ),
        }

    def evaluate(self, prediction: dict, outcome: dict) -> dict:
        predicted_stops = prediction.get("value", 1)
        winner_stops = outcome.get("winner_stops", 1)
        modal_stops = outcome.get("most_common_stops", 1)

        correct = predicted_stops == modal_stops

        return {
            "correct": correct,
            "predicted_stops": predicted_stops,
            "winner_strategy": winner_stops,
            "field_strategy": modal_stops,
            "details": f"Predicted {predicted_stops}-stop, winner did {winner_stops}-stop, field mostly {modal_stops}-stop",
            "score": 1.0 if correct else 0.0,
        }

    def required_data(self) -> list[str]:
        return ["circuitId", "race_laps"]
