"""Race weekend prediction runner.

Orchestrates all hypotheses for an upcoming race: gathers pre-race data,
runs every hypothesis, records predictions, and generates a prediction
report. After the race, resolves predictions and updates the scorecard.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config import PROJECT_ROOT
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
from .registry import PredictionRegistry
from .scorecard import Scorecard

logger = logging.getLogger(__name__)

REPORTS_DIR = PROJECT_ROOT / "predictions" / "reports"


class RacePredictor:
    """Generate and manage predictions for a race weekend.

    Usage:
        predictor = RacePredictor(historical_results=results_df)

        # Before the race
        report = predictor.predict_race(2025, 1, "Bahrain", pre_race_data)

        # After the race
        predictor.resolve_race(2025, 1, outcomes)

        # Check how we're doing
        predictor.scorecard.plot_dashboard()
    """

    def __init__(
        self,
        historical_results: Optional[pd.DataFrame] = None,
        registry: Optional[PredictionRegistry] = None,
    ):
        self.registry = registry or PredictionRegistry()
        self.scorecard = Scorecard(self.registry)

        # Initialize all hypotheses
        self.hypotheses: dict[str, Hypothesis] = {
            "pole_winner": PoleWinnerHypothesis(historical_results),
            "retirement_risk": RetirementRiskHypothesis(historical_results),
            "safety_car": SafetyCarHypothesis(),
            "undercut_effective": UndercutHypothesis(),
            "wet_weather_upset": WetWeatherUpsetHypothesis(),
            "first_lap_incident": FirstLapIncidentHypothesis(),
            "tire_strategy": TireStrategyHypothesis(),
        }

    def predict_race(
        self,
        season: int,
        round_num: int,
        gp_name: str,
        pre_race_data: dict,
    ) -> dict:
        """Run all hypotheses for an upcoming race.

        Args:
            season: Season year.
            round_num: Round number.
            gp_name: Grand Prix name.
            pre_race_data: Dict with circuit info, qualifying results,
                weather forecast, etc.

        Returns:
            Dict mapping hypothesis names to their predictions.
        """
        predictions = {}

        for name, hypothesis in self.hypotheses.items():
            try:
                prediction = hypothesis.predict(pre_race_data)

                self.registry.record(
                    hypothesis_name=name,
                    season=season,
                    round_num=round_num,
                    gp_name=gp_name,
                    prediction=prediction,
                    confidence=prediction.get("confidence", 0.5),
                    reasoning=prediction.get("reasoning", ""),
                )

                predictions[name] = prediction
                logger.info(
                    "[%s] %s: %s (confidence=%.0f%%)",
                    gp_name, name,
                    prediction.get("value", "?"),
                    prediction.get("confidence", 0) * 100,
                )

            except Exception as e:
                logger.warning("Hypothesis %s failed for %s: %s", name, gp_name, e)

        # Save human-readable report
        self._save_report(season, round_num, gp_name, predictions)

        return predictions

    def resolve_race(
        self,
        season: int,
        round_num: int,
        outcomes: dict[str, dict],
    ) -> dict:
        """Resolve all predictions after a race completes.

        Args:
            outcomes: Dict mapping hypothesis name to outcome dict.
                      e.g. {"pole_winner": {"winner_from_pole": True}, ...}

        Returns:
            Summary of what was correct/incorrect.
        """
        summary = {}

        for name, outcome in outcomes.items():
            resolved = self.registry.resolve(season, round_num, name, outcome)
            for pred in resolved:
                summary[name] = {
                    "correct": pred["correct"],
                    "prediction": pred["prediction"],
                    "outcome": pred["outcome"],
                }

        logger.info(
            "Resolved R%d: %d/%d correct",
            round_num,
            sum(1 for v in summary.values() if v.get("correct")),
            len(summary),
        )

        return summary

    def build_pre_race_data(
        self,
        qualifying_results: pd.DataFrame,
        circuit_id: str,
        race_laps: int = 55,
        weather_forecast: Optional[dict] = None,
    ) -> dict:
        """Helper to build pre_race_data dict from available data.

        Args:
            qualifying_results: Qualifying results DataFrame.
            circuit_id: Circuit identifier string.
            race_laps: Number of laps in the race.
            weather_forecast: Optional weather dict with rain_probability, etc.
        """
        data = {
            "circuitId": circuit_id,
            "race_laps": race_laps,
        }

        # Extract pole sitter
        if not qualifying_results.empty:
            if "position" in qualifying_results.columns:
                pole = qualifying_results[qualifying_results["position"] == 1]
                if not pole.empty:
                    data["pole_driver"] = pole.iloc[0].get("driver_code", "unknown")

            # Driver list
            drivers = []
            for _, row in qualifying_results.iterrows():
                drivers.append({
                    "driverId": row.get("driverId", ""),
                    "constructorId": row.get("constructorId", ""),
                    "driver_code": row.get("driver_code", ""),
                })
            data["drivers"] = drivers

            # Qualifying spread
            if "Q3" in qualifying_results.columns:
                q3_times = pd.to_timedelta(qualifying_results["Q3"].dropna())
                if len(q3_times) >= 2:
                    spread = (q3_times.max() - q3_times.min()).total_seconds()
                    data["quali_spread_seconds"] = spread

        # Weather
        if weather_forecast:
            data.update(weather_forecast)

        # Street circuit detection
        street_circuits = {"monaco", "jeddah", "baku", "singapore", "las_vegas"}
        data["is_street_circuit"] = circuit_id in street_circuits

        return data

    def build_post_race_outcomes(
        self,
        race_results: pd.DataFrame,
        pit_stops: Optional[pd.DataFrame] = None,
        had_safety_car: bool = False,
        had_rain: bool = False,
    ) -> dict[str, dict]:
        """Helper to build outcomes dict from post-race data.

        Args:
            race_results: Race results DataFrame.
            pit_stops: Pit stop data.
            had_safety_car: Whether safety car was deployed.
            had_rain: Whether it rained during the race.
        """
        outcomes: dict[str, dict] = {}

        # Pole winner
        pole = race_results[race_results["grid"] == 1]
        winner = race_results[race_results["position"] == 1]
        if not pole.empty and not winner.empty:
            pole_won = pole["driverId"].iloc[0] == winner["driverId"].iloc[0]
            outcomes["pole_winner"] = {
                "winner_from_pole": pole_won,
                "value": pole_won,
            }

        # Retirements
        finished_statuses = {"Finished", "+1 Lap", "+2 Laps", "+3 Laps"}
        if "status" in race_results.columns:
            dnfs = race_results[~race_results["status"].isin(finished_statuses)]
            outcomes["retirement_risk"] = {
                "dnf_drivers": dnfs["driverId"].tolist(),
                "dnf_count": len(dnfs),
                "values": dnfs["driverId"].tolist(),
            }

        # Safety car
        outcomes["safety_car"] = {
            "safety_car": had_safety_car,
            "occurred": had_safety_car,
        }

        # Wet weather upset
        if "grid" in race_results.columns and "position" in race_results.columns:
            df = race_results.dropna(subset=["grid", "position"])
            big_movers = df[(df["grid"] - df["position"]) >= 10]
            non_top5_winner = not winner.empty and winner["grid"].iloc[0] > 5
            upset = len(big_movers) > 0 or non_top5_winner

            outcomes["wet_weather_upset"] = {
                "upset_occurred": upset and had_rain,
                "occurred": upset and had_rain,
                "upset_details": f"{len(big_movers)} drivers gained 10+ places"
                if big_movers is not None else "",
            }

        # First lap incident (simplified — would need race control data)
        outcomes["first_lap_incident"] = {
            "first_lap_incident": False,  # needs manual or RC data input
        }

        # Tire strategy
        if pit_stops is not None and not pit_stops.empty:
            stop_counts = pit_stops.groupby("driverId")["stop"].max()
            modal_stops = int(stop_counts.mode().iloc[0]) if len(stop_counts) > 0 else 1
            winner_id = winner["driverId"].iloc[0] if not winner.empty else None
            winner_stops = int(stop_counts.get(winner_id, 1)) if winner_id else 1

            outcomes["tire_strategy"] = {
                "winner_stops": winner_stops,
                "most_common_stops": modal_stops,
                "values": [str(modal_stops)],
            }

        # Undercut (simplified)
        outcomes["undercut_effective"] = {
            "undercut_gained_position": False,  # needs detailed analysis
        }

        return outcomes

    def _save_report(
        self,
        season: int,
        round_num: int,
        gp_name: str,
        predictions: dict,
    ) -> Path:
        """Save a human-readable prediction report."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / f"{season}_R{round_num:02d}_{gp_name.replace(' ', '_')}.md"

        lines = [
            f"# {season} {gp_name} — Pre-Race Predictions",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]

        # Overall accuracy so far
        summary = self.scorecard.summary()
        if not summary.empty:
            total_correct = summary["correct_count"].sum()
            total_preds = summary["total"].sum()
            lines.append(
                f"**Season track record: {total_correct}/{total_preds} "
                f"({total_correct/total_preds:.0%})**" if total_preds > 0 else ""
            )
            lines.append("")

        for name, pred in predictions.items():
            confidence = pred.get("confidence", 0)
            conf_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))

            lines.extend([
                f"## {name.replace('_', ' ').title()}",
                f"**Prediction:** {pred.get('value', '?')}",
                f"**Confidence:** {conf_bar} {confidence:.0%}",
                f"**Reasoning:** {pred.get('reasoning', 'N/A')}",
                "",
            ])

            # Add any extra fields
            for key in ("probability", "strategy_description", "expected_dnfs", "risk_ranking"):
                if key in pred:
                    val = pred[key]
                    if key == "risk_ranking" and isinstance(val, list):
                        lines.append(f"**Risk ranking (top 3):**")
                        for entry in val[:3]:
                            lines.append(
                                f"  - {entry.get('driverId', '?')}: "
                                f"{entry.get('dnf_probability', 0):.0%}"
                            )
                    else:
                        lines.append(f"**{key}:** {val}")
                    lines.append("")

            lines.append("---")
            lines.append("")

        path.write_text("\n".join(lines))
        logger.info("Report saved to %s", path)
        return path

    def season_report(self) -> str:
        """Generate a text summary of season prediction performance."""
        summary = self.scorecard.summary()
        if summary.empty:
            return "No predictions have been resolved yet."

        lines = ["# Season Prediction Performance", ""]

        total_correct = int(summary["correct_count"].sum())
        total_preds = int(summary["total"].sum())
        overall = total_correct / total_preds if total_preds > 0 else 0

        lines.append(f"**Overall: {total_correct}/{total_preds} ({overall:.0%})**")
        lines.append("")
        lines.append("| Hypothesis | Accuracy | N | Avg Confidence |")
        lines.append("|------------|----------|---|----------------|")

        for name, row in summary.iterrows():
            lines.append(
                f"| {name} | {row['accuracy']:.0%} | {int(row['total'])} | {row['avg_confidence']:.0%} |"
            )

        lines.append("")

        # Best and worst
        if len(summary) > 1:
            best = summary["accuracy"].idxmax()
            worst = summary["accuracy"].idxmin()
            lines.append(f"**Strongest hypothesis:** {best} ({summary.loc[best, 'accuracy']:.0%})")
            lines.append(f"**Weakest hypothesis:** {worst} ({summary.loc[worst, 'accuracy']:.0%})")

        return "\n".join(lines)
