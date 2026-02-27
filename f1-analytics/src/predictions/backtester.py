"""Backtesting engine — validate hypotheses against historical data.

Runs each hypothesis through past seasons to build a track record before
using it for live predictions. This is how we know which hypotheses are
worth listening to and which need more work.
"""

import logging
from typing import Optional

import pandas as pd

from .hypotheses import Hypothesis, PoleWinnerHypothesis, RetirementRiskHypothesis
from .registry import PredictionRegistry

logger = logging.getLogger(__name__)


class Backtester:
    """Run hypotheses through historical data and score them."""

    def __init__(self, registry: Optional[PredictionRegistry] = None):
        self.registry = registry or PredictionRegistry()

    def backtest_pole_winner(
        self,
        results: pd.DataFrame,
        train_end_year: int = 2022,
    ) -> pd.DataFrame:
        """Backtest the pole-winner hypothesis.

        Uses data up to train_end_year to fit the model, then predicts
        each subsequent race and scores it.

        Args:
            results: Historical race results (from JolpicaClient).
            train_end_year: Last year used for training.

        Returns:
            DataFrame of prediction results.
        """
        train = results[results["season"] <= train_end_year]
        test = results[results["season"] > train_end_year]

        hypothesis = PoleWinnerHypothesis(historical_results=train)
        return self._run(hypothesis, test, "pole_winner")

    def backtest_retirement_risk(
        self,
        results: pd.DataFrame,
        train_end_year: int = 2022,
    ) -> pd.DataFrame:
        """Backtest the retirement risk hypothesis."""
        train = results[results["season"] <= train_end_year]
        test = results[results["season"] > train_end_year]

        hypothesis = RetirementRiskHypothesis(historical_results=train)
        return self._run(hypothesis, test, "retirement_risk")

    def _run(
        self,
        hypothesis: Hypothesis,
        test_results: pd.DataFrame,
        name: str,
    ) -> pd.DataFrame:
        """Run a hypothesis through test data race by race."""
        rows = []

        for (season, rnd), race in test_results.groupby(["season", "round"]):
            # Build pre_race_data from the race's grid
            pre_race_data = self._build_pre_race_data(race)

            # Generate prediction
            prediction = hypothesis.predict(pre_race_data)

            # Build outcome from actual results
            outcome = self._build_outcome(race, name)

            # Evaluate
            evaluation = hypothesis.evaluate(prediction, outcome)

            # Record in registry
            gp_name = race["raceName"].iloc[0] if "raceName" in race.columns else f"R{rnd}"
            self.registry.record(
                hypothesis_name=name,
                season=int(season),
                round_num=int(rnd),
                gp_name=gp_name,
                prediction=prediction,
                confidence=prediction.get("confidence", 0.5),
                reasoning=prediction.get("reasoning", ""),
                model_version="backtest_v1",
            )
            self.registry.resolve(int(season), int(rnd), name, outcome)

            rows.append({
                "season": season,
                "round": rnd,
                "gp_name": gp_name,
                "prediction": prediction,
                "outcome": outcome,
                "correct": evaluation.get("correct"),
                "score": evaluation.get("score", 0),
                "confidence": prediction.get("confidence", 0.5),
                "details": evaluation.get("details", ""),
            })

        return pd.DataFrame(rows)

    @staticmethod
    def _build_pre_race_data(race: pd.DataFrame) -> dict:
        """Build pre-race data dict from a race's results DataFrame."""
        data: dict = {}

        if "circuitId" in race.columns:
            data["circuitId"] = race["circuitId"].iloc[0]

        # Pole sitter
        pole_row = race[race["grid"] == 1]
        if not pole_row.empty:
            data["pole_driver"] = pole_row["driver_code"].iloc[0]

        # Driver list
        drivers = []
        for _, row in race.iterrows():
            drivers.append({
                "driverId": row.get("driverId", ""),
                "constructorId": row.get("constructorId", ""),
                "driver_code": row.get("driver_code", ""),
            })
        data["drivers"] = drivers

        return data

    @staticmethod
    def _build_outcome(race: pd.DataFrame, hypothesis_name: str) -> dict:
        """Build an outcome dict from actual race results."""
        outcome: dict = {}

        if hypothesis_name == "pole_winner":
            pole_row = race[race["grid"] == 1]
            winner_row = race[race["position"] == 1]
            if not pole_row.empty and not winner_row.empty:
                outcome["winner_from_pole"] = (
                    pole_row["driverId"].iloc[0] == winner_row["driverId"].iloc[0]
                )
                outcome["value"] = outcome["winner_from_pole"]
            else:
                outcome["winner_from_pole"] = False
                outcome["value"] = False

        elif hypothesis_name == "retirement_risk":
            dnf_statuses = {"Finished", "+1 Lap", "+2 Laps", "+3 Laps"}
            if "status" in race.columns:
                dnf_drivers = race[~race["status"].isin(dnf_statuses)]["driverId"].tolist()
            else:
                dnf_drivers = []
            outcome["dnf_drivers"] = dnf_drivers
            outcome["dnf_count"] = len(dnf_drivers)
            outcome["values"] = dnf_drivers

        return outcome

    def run_all(
        self,
        results: pd.DataFrame,
        train_end_year: int = 2022,
    ) -> dict[str, pd.DataFrame]:
        """Run all available backtests.

        Returns dict mapping hypothesis name to results DataFrame.
        """
        all_results = {}

        logger.info("Backtesting pole_winner...")
        all_results["pole_winner"] = self.backtest_pole_winner(results, train_end_year)

        logger.info("Backtesting retirement_risk...")
        all_results["retirement_risk"] = self.backtest_retirement_risk(results, train_end_year)

        # Print summary
        for name, df in all_results.items():
            if not df.empty:
                accuracy = df["correct"].mean()
                avg_score = df["score"].mean()
                logger.info(
                    "%s: accuracy=%.1f%%, avg_score=%.3f, n=%d",
                    name, accuracy * 100, avg_score, len(df),
                )

        return all_results
