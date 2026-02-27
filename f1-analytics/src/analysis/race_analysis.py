"""Race outcome analysis.

Analyzes qualifying-to-race correlation, overtaking difficulty per circuit,
and pit stop strategy effectiveness.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class RaceAnalyzer:
    """Statistical analysis of race outcomes and strategy."""

    # ------------------------------------------------------------------
    # Qualifying vs race correlation
    # ------------------------------------------------------------------

    @staticmethod
    def qualifying_race_correlation(results: pd.DataFrame) -> dict:
        """Compute correlation between grid position and finishing position.

        Args:
            results: DataFrame with 'grid' and 'position' columns.

        Returns:
            Dict with Spearman and Pearson correlations.
        """
        df = results.dropna(subset=["grid", "position"])
        df = df[(df["grid"] > 0) & (df["position"] > 0)]

        if len(df) < 3:
            return {"spearman_r": np.nan, "pearson_r": np.nan, "n": 0}

        spearman_r, spearman_p = stats.spearmanr(df["grid"], df["position"])
        pearson_r, pearson_p = stats.pearsonr(df["grid"], df["position"])

        return {
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "n": len(df),
        }

    @staticmethod
    def correlation_by_circuit(results: pd.DataFrame) -> pd.DataFrame:
        """Compute qualifying-race correlation per circuit.

        Quantifies 'overtaking difficulty': high correlation = hard to overtake.
        """
        rows = []
        for circuit, group in results.groupby("circuitId"):
            corr = RaceAnalyzer.qualifying_race_correlation(group)
            corr["circuitId"] = circuit
            corr["avg_position_change"] = (group["grid"] - group["position"]).abs().mean()
            rows.append(corr)

        df = pd.DataFrame(rows).sort_values("spearman_r", ascending=False)
        return df

    # ------------------------------------------------------------------
    # Position changes analysis
    # ------------------------------------------------------------------

    @staticmethod
    def position_changes(results: pd.DataFrame) -> pd.DataFrame:
        """Compute positions gained/lost per driver per race."""
        df = results.dropna(subset=["grid", "position"]).copy()
        df["positions_gained"] = df["grid"] - df["position"]
        return df

    @staticmethod
    def overtaking_difficulty_index(results: pd.DataFrame) -> pd.DataFrame:
        """Rank circuits by how often the grid order is preserved.

        A higher index means grid position is more predictive of the result
        (harder to overtake).
        """
        df = RaceAnalyzer.position_changes(results)
        circuit_stats = (
            df.groupby("circuitId")
            .agg(
                avg_abs_change=("positions_gained", lambda x: x.abs().mean()),
                grid_winner_pct=("positions_gained", lambda x: (x.iloc[:1] >= 0).mean()),
                n_races=("positions_gained", "count"),
            )
            .sort_values("avg_abs_change")
        )
        return circuit_stats

    # ------------------------------------------------------------------
    # Pit strategy analysis
    # ------------------------------------------------------------------

    @staticmethod
    def analyze_pit_strategy(
        pit_stops: pd.DataFrame, results: pd.DataFrame
    ) -> pd.DataFrame:
        """Correlate number of pit stops with race outcome.

        Args:
            pit_stops: DataFrame with pit stop data (driverId, stop, lap).
            results: Race results DataFrame.

        Returns:
            Merged DataFrame with stop counts and finishing positions.
        """
        stop_counts = (
            pit_stops.groupby(["season", "round", "driverId"])["stop"]
            .max()
            .reset_index()
            .rename(columns={"stop": "num_stops"})
        )

        merged = results.merge(stop_counts, on=["season", "round", "driverId"], how="left")
        merged["num_stops"] = merged["num_stops"].fillna(0).astype(int)
        return merged

    @staticmethod
    def undercut_overcut_analysis(
        pit_stops: pd.DataFrame,
        laps: pd.DataFrame,
    ) -> pd.DataFrame:
        """Analyze the effectiveness of undercutting and overcutting.

        Compares position changes around pit stop windows between drivers
        who pit first (undercut) vs. later (overcut).
        """
        if pit_stops.empty or laps.empty:
            return pd.DataFrame()

        # Group pit stops by race and find pairs of drivers with close stop laps
        results = []
        for (season, rnd), group in pit_stops.groupby(["season", "round"]):
            first_stops = group[group["stop"] == 1].sort_values("lap")
            for i in range(len(first_stops)):
                for j in range(i + 1, len(first_stops)):
                    a = first_stops.iloc[i]
                    b = first_stops.iloc[j]
                    lap_diff = b["lap"] - a["lap"]

                    if 1 <= lap_diff <= 5:
                        results.append({
                            "season": season,
                            "round": rnd,
                            "undercut_driver": a["driverId"],
                            "overcut_driver": b["driverId"],
                            "undercut_lap": a["lap"],
                            "overcut_lap": b["lap"],
                            "lap_difference": lap_diff,
                        })

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    @staticmethod
    def plot_grid_vs_finish(
        results: pd.DataFrame,
        circuit: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Scatter plot of grid position vs finishing position."""
        df = results.dropna(subset=["grid", "position"])
        if circuit:
            df = df[df["circuitId"] == circuit]

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure

        ax.scatter(df["grid"], df["position"], alpha=0.3, s=20)
        max_pos = max(df["grid"].max(), df["position"].max())
        ax.plot([1, max_pos], [1, max_pos], "r--", alpha=0.5, label="No change")

        corr = RaceAnalyzer.qualifying_race_correlation(df)
        ax.set_xlabel("Grid Position")
        ax.set_ylabel("Finishing Position")
        title = f"Grid vs Finish — {circuit}" if circuit else "Grid vs Finish"
        ax.set_title(f"{title}\n(Spearman r={corr['spearman_r']:.3f})")
        ax.legend()
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_overtaking_difficulty(
        results: pd.DataFrame, top_n: int = 15, ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """Bar chart of circuits ranked by overtaking difficulty."""
        odi = RaceAnalyzer.overtaking_difficulty_index(results)
        odi = odi.head(top_n)

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.figure

        ax.barh(odi.index, odi["avg_abs_change"])
        ax.set_xlabel("Avg Absolute Position Change")
        ax.set_title("Overtaking Difficulty by Circuit (lower = harder)")
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_strategy_outcomes(
        strategy_df: pd.DataFrame, ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """Box plot of finishing position by number of pit stops."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        sns.boxplot(data=strategy_df, x="num_stops", y="position", ax=ax)
        ax.set_xlabel("Number of Pit Stops")
        ax.set_ylabel("Finishing Position")
        ax.set_title("Race Outcome by Pit Strategy")
        fig.tight_layout()
        return fig
