"""Prediction registry — persistent store for predictions and outcomes.

Every prediction gets a unique ID, is stored as JSON, and can be resolved
against actual outcomes. The registry accumulates over time, building a
track record that reveals which hypotheses are predictive and which aren't.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config import PROJECT_ROOT

logger = logging.getLogger(__name__)

PREDICTIONS_DIR = PROJECT_ROOT / "predictions" / "history"


class PredictionRegistry:
    """Store, resolve, and query predictions."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or PREDICTIONS_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _race_file(self, season: int, round_num: int) -> Path:
        return self.storage_dir / f"{season}_R{round_num:02d}.json"

    def _load_race(self, season: int, round_num: int) -> list[dict]:
        path = self._race_file(season, round_num)
        if path.exists():
            return json.loads(path.read_text())
        return []

    def _save_race(self, season: int, round_num: int, predictions: list[dict]) -> None:
        path = self._race_file(season, round_num)
        path.write_text(json.dumps(predictions, indent=2, default=str))

    def record(
        self,
        hypothesis_name: str,
        season: int,
        round_num: int,
        gp_name: str,
        prediction: dict,
        confidence: float,
        reasoning: str,
        model_version: str = "v1",
    ) -> str:
        """Record a new prediction before a race.

        Args:
            hypothesis_name: Which hypothesis generated this (e.g. 'pole_winner').
            season: Season year.
            round_num: Round number.
            gp_name: Grand Prix name.
            prediction: The actual prediction content (varies by hypothesis).
            confidence: Model confidence 0.0–1.0.
            reasoning: Human-readable explanation of why this prediction was made.
            model_version: Version tag for the model/logic that produced it.

        Returns:
            Unique prediction ID.
        """
        pred_id = str(uuid.uuid4())[:8]

        entry = {
            "id": pred_id,
            "hypothesis": hypothesis_name,
            "season": season,
            "round": round_num,
            "gp_name": gp_name,
            "prediction": prediction,
            "confidence": confidence,
            "reasoning": reasoning,
            "model_version": model_version,
            "created_at": datetime.now().isoformat(),
            "outcome": None,
            "correct": None,
            "resolved_at": None,
        }

        predictions = self._load_race(season, round_num)
        predictions.append(entry)
        self._save_race(season, round_num, predictions)

        logger.info(
            "Recorded prediction %s: %s for %s R%d (confidence=%.2f)",
            pred_id, hypothesis_name, gp_name, round_num, confidence,
        )
        return pred_id

    def resolve(
        self,
        season: int,
        round_num: int,
        hypothesis_name: str,
        outcome: dict,
    ) -> list[dict]:
        """Resolve predictions against actual outcomes.

        Finds all unresolved predictions for this race and hypothesis,
        calls the hypothesis-specific scoring logic, and marks them.

        Args:
            outcome: The actual race outcome data.

        Returns:
            List of resolved prediction entries.
        """
        predictions = self._load_race(season, round_num)
        resolved = []

        for pred in predictions:
            if pred["hypothesis"] == hypothesis_name and pred["outcome"] is None:
                pred["outcome"] = outcome
                pred["correct"] = self._evaluate(pred["prediction"], outcome)
                pred["resolved_at"] = datetime.now().isoformat()
                resolved.append(pred)

        self._save_race(season, round_num, predictions)
        logger.info("Resolved %d predictions for %s R%d", len(resolved), hypothesis_name, round_num)
        return resolved

    @staticmethod
    def _evaluate(prediction: dict, outcome: dict) -> Optional[bool]:
        """Basic evaluation — checks if the predicted value matches outcome.

        Supports different prediction types:
        - binary: prediction['value'] == outcome['value']
        - categorical: prediction['value'] in outcome['values']
        - ranked: measures positional accuracy
        """
        pred_type = prediction.get("type", "binary")

        if pred_type == "binary":
            return prediction.get("value") == outcome.get("value")

        if pred_type == "categorical":
            return prediction.get("value") in outcome.get("values", [])

        if pred_type == "probability":
            # For probability predictions, correct if the predicted outcome occurred
            return outcome.get("occurred", False)

        if pred_type == "top_n":
            predicted_set = set(prediction.get("values", []))
            actual_set = set(outcome.get("values", []))
            return len(predicted_set & actual_set) > 0

        return None

    def get_all(self) -> pd.DataFrame:
        """Load all predictions across all races into a DataFrame."""
        rows = []
        for path in sorted(self.storage_dir.glob("*.json")):
            entries = json.loads(path.read_text())
            rows.extend(entries)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        return df

    def get_race(self, season: int, round_num: int) -> pd.DataFrame:
        """Get all predictions for a specific race."""
        entries = self._load_race(season, round_num)
        return pd.DataFrame(entries) if entries else pd.DataFrame()

    def accuracy_by_hypothesis(self) -> pd.DataFrame:
        """Compute accuracy stats grouped by hypothesis type."""
        df = self.get_all()
        if df.empty or "correct" not in df.columns:
            return pd.DataFrame()

        resolved = df[df["correct"].notna()].copy()
        resolved["correct"] = resolved["correct"].astype(bool)

        return (
            resolved.groupby("hypothesis")
            .agg(
                total=("correct", "count"),
                correct_count=("correct", "sum"),
                accuracy=("correct", "mean"),
                avg_confidence=("confidence", "mean"),
            )
            .sort_values("accuracy", ascending=False)
        )

    def accuracy_over_time(self) -> pd.DataFrame:
        """Track prediction accuracy across races (cumulative)."""
        df = self.get_all()
        if df.empty:
            return pd.DataFrame()

        resolved = df[df["correct"].notna()].copy()
        resolved["correct"] = resolved["correct"].astype(bool)
        resolved = resolved.sort_values("created_at")

        resolved["cumulative_correct"] = resolved["correct"].cumsum()
        resolved["cumulative_total"] = range(1, len(resolved) + 1)
        resolved["cumulative_accuracy"] = resolved["cumulative_correct"] / resolved["cumulative_total"]

        return resolved
