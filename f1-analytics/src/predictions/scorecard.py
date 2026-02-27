"""Prediction scorecard — tracks and visualizes prediction accuracy over time.

The scorecard is the feedback loop: it shows which hypotheses are working,
which need recalibration, and how overall prediction quality evolves.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .registry import PredictionRegistry

logger = logging.getLogger(__name__)


class Scorecard:
    """Generate accuracy reports and calibration diagnostics."""

    def __init__(self, registry: PredictionRegistry):
        self.registry = registry

    def summary(self) -> pd.DataFrame:
        """One-line-per-hypothesis accuracy summary."""
        return self.registry.accuracy_by_hypothesis()

    def calibration_table(self, n_bins: int = 5) -> pd.DataFrame:
        """Check if confidence scores are well calibrated.

        Groups predictions into confidence bins and compares predicted
        confidence with actual accuracy in each bin. A well-calibrated
        model has predicted_confidence ≈ actual_accuracy in every bin.
        """
        df = self.registry.get_all()
        if df.empty:
            return pd.DataFrame()

        resolved = df[df["correct"].notna()].copy()
        resolved["correct"] = resolved["correct"].astype(float)

        # Bin by confidence
        resolved["conf_bin"] = pd.cut(resolved["confidence"], bins=n_bins)

        cal = (
            resolved.groupby("conf_bin", observed=True)
            .agg(
                mean_confidence=("confidence", "mean"),
                actual_accuracy=("correct", "mean"),
                count=("correct", "count"),
            )
        )
        cal["calibration_error"] = abs(cal["mean_confidence"] - cal["actual_accuracy"])
        return cal

    def hypothesis_trend(self, hypothesis_name: str, window: int = 5) -> pd.DataFrame:
        """Rolling accuracy for a specific hypothesis.

        Shows whether the hypothesis is getting better or worse over time.
        """
        df = self.registry.get_all()
        if df.empty:
            return pd.DataFrame()

        hyp = df[df["hypothesis"] == hypothesis_name].copy()
        hyp = hyp[hyp["correct"].notna()].sort_values("created_at")
        hyp["correct"] = hyp["correct"].astype(float)
        hyp["rolling_accuracy"] = hyp["correct"].rolling(window, min_periods=1).mean()

        return hyp

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_dashboard(self) -> plt.Figure:
        """Full scorecard dashboard with multiple panels."""
        fig = plt.figure(figsize=(18, 14))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

        summary = self.summary()
        if summary.empty:
            fig.text(0.5, 0.5, "No resolved predictions yet", ha="center", fontsize=16)
            return fig

        # Panel 1: Accuracy by hypothesis
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_accuracy_bars(summary, ax1)

        # Panel 2: Calibration plot
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_calibration(ax2)

        # Panel 3: Cumulative accuracy over time
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_cumulative_accuracy(ax3)

        # Panel 4: Confidence vs accuracy scatter
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_confidence_scatter(ax4)

        # Panel 5: Per-hypothesis trends
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_hypothesis_trends(ax5)

        fig.suptitle("F1 Prediction Scorecard", fontsize=16)
        return fig

    def _plot_accuracy_bars(self, summary: pd.DataFrame, ax: plt.Axes) -> None:
        colors = ["#2ecc71" if acc > 0.6 else "#e74c3c" if acc < 0.4 else "#f39c12"
                   for acc in summary["accuracy"]]
        ax.barh(summary.index, summary["accuracy"], color=colors)
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Accuracy")
        ax.set_title("Accuracy by Hypothesis")

        for i, (acc, n) in enumerate(zip(summary["accuracy"], summary["total"])):
            ax.text(acc + 0.02, i, f"{acc:.0%} (n={n})", va="center", fontsize=9)

    def _plot_calibration(self, ax: plt.Axes) -> None:
        cal = self.calibration_table()
        if cal.empty:
            ax.text(0.5, 0.5, "Insufficient data", ha="center")
            return

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.scatter(
            cal["mean_confidence"], cal["actual_accuracy"],
            s=cal["count"] * 20, alpha=0.7, c="#3498db",
        )
        ax.set_xlabel("Predicted Confidence")
        ax.set_ylabel("Actual Accuracy")
        ax.set_title("Calibration Plot")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend()

    def _plot_cumulative_accuracy(self, ax: plt.Axes) -> None:
        df = self.registry.accuracy_over_time()
        if df.empty:
            return

        ax.plot(df["cumulative_total"], df["cumulative_accuracy"], linewidth=2)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.fill_between(
            df["cumulative_total"], df["cumulative_accuracy"], 0.5,
            where=df["cumulative_accuracy"] > 0.5, alpha=0.2, color="green",
        )
        ax.fill_between(
            df["cumulative_total"], df["cumulative_accuracy"], 0.5,
            where=df["cumulative_accuracy"] < 0.5, alpha=0.2, color="red",
        )
        ax.set_xlabel("Total Predictions Resolved")
        ax.set_ylabel("Cumulative Accuracy")
        ax.set_title("Prediction Accuracy Over Time")

    def _plot_confidence_scatter(self, ax: plt.Axes) -> None:
        df = self.registry.get_all()
        if df.empty:
            return

        resolved = df[df["correct"].notna()].copy()
        resolved["correct"] = resolved["correct"].astype(int)

        ax.scatter(
            resolved["confidence"],
            resolved["correct"] + np.random.normal(0, 0.03, len(resolved)),
            alpha=0.4, s=20,
        )
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Correct (jittered)")
        ax.set_title("Confidence vs Outcome")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Wrong", "Correct"])

    def _plot_hypothesis_trends(self, ax: plt.Axes) -> None:
        df = self.registry.get_all()
        if df.empty:
            return

        for hyp_name in df["hypothesis"].unique():
            trend = self.hypothesis_trend(hyp_name)
            if len(trend) >= 3:
                ax.plot(
                    range(len(trend)), trend["rolling_accuracy"].values,
                    label=hyp_name, marker=".", markersize=4,
                )

        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Prediction #")
        ax.set_ylabel("Rolling Accuracy")
        ax.set_title("Hypothesis Accuracy Trends")
        ax.legend(fontsize=8, loc="lower right")
