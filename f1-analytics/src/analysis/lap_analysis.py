"""Lap time statistical analysis.

Compares lap time distributions across drivers, sessions, and tire compounds.
Includes tire degradation modeling and statistical significance testing.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class LapAnalyzer:
    """Statistical analysis of lap times."""

    @staticmethod
    def filter_representative_laps(
        laps: pd.DataFrame,
        exclude_pit_laps: bool = True,
        max_delta_pct: float = 1.07,
    ) -> pd.DataFrame:
        """Filter out non-representative laps (pit in/out, outliers, SC laps).

        Args:
            laps: Laps DataFrame.
            exclude_pit_laps: Remove pit in/out laps.
            max_delta_pct: Remove laps slower than this fraction of the median.
        """
        df = laps.copy()

        if exclude_pit_laps:
            for col in ["PitInTime", "PitOutTime"]:
                if col in df.columns:
                    df = df[df[col].isna()]

        if "LapTime" in df.columns:
            if hasattr(df["LapTime"], "dt"):
                seconds = df["LapTime"].dt.total_seconds()
            else:
                seconds = df["LapTime"]
            median = seconds.median()
            df = df[seconds <= median * max_delta_pct]

        return df

    # ------------------------------------------------------------------
    # Distribution comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_distributions(
        laps: pd.DataFrame,
        group_col: str = "Driver",
        time_col: str = "LapTime",
    ) -> pd.DataFrame:
        """Compute summary statistics per group (driver, compound, etc.).

        Returns DataFrame with mean, median, std, and count per group.
        """
        df = laps.copy()
        if hasattr(df[time_col], "dt"):
            df["_seconds"] = df[time_col].dt.total_seconds()
        else:
            df["_seconds"] = df[time_col]

        summary = (
            df.groupby(group_col)["_seconds"]
            .agg(["mean", "median", "std", "count"])
            .sort_values("median")
        )
        return summary

    @staticmethod
    def mann_whitney_test(
        group_a: pd.Series, group_b: pd.Series
    ) -> dict:
        """Two-sample Mann-Whitney U test for lap time difference."""
        a = group_a.dt.total_seconds() if hasattr(group_a, "dt") else group_a
        b = group_b.dt.total_seconds() if hasattr(group_b, "dt") else group_b
        stat, p = stats.mannwhitneyu(a.dropna(), b.dropna(), alternative="two-sided")
        return {"U_statistic": stat, "p_value": p, "significant_005": p < 0.05}

    @staticmethod
    def kruskal_wallis_test(
        laps: pd.DataFrame,
        group_col: str = "Driver",
        time_col: str = "LapTime",
    ) -> dict:
        """Kruskal-Wallis H-test across multiple groups."""
        df = laps.copy()
        if hasattr(df[time_col], "dt"):
            df["_seconds"] = df[time_col].dt.total_seconds()
        else:
            df["_seconds"] = df[time_col]

        groups = [g["_seconds"].dropna().values for _, g in df.groupby(group_col)]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            return {"H_statistic": np.nan, "p_value": np.nan, "significant_005": False}

        stat, p = stats.kruskal(*groups)
        return {"H_statistic": stat, "p_value": p, "significant_005": p < 0.05}

    # ------------------------------------------------------------------
    # Tire degradation
    # ------------------------------------------------------------------

    @staticmethod
    def fit_degradation_curve(
        laps: pd.DataFrame,
        degree: int = 2,
        time_col: str = "LapTime",
        stint_lap_col: str = "TyreLife",
    ) -> dict:
        """Fit a polynomial degradation model (lap time vs. tire age).

        Args:
            laps: Filtered laps for a single stint/compound.
            degree: Polynomial degree (1=linear, 2=quadratic).
            time_col: Lap time column.
            stint_lap_col: Tire life column (laps on current set).

        Returns:
            Dict with coefficients, predictions, and R² score.
        """
        df = laps.dropna(subset=[time_col, stint_lap_col]).copy()
        if hasattr(df[time_col], "dt"):
            y = df[time_col].dt.total_seconds().values
        else:
            y = df[time_col].values
        x = df[stint_lap_col].values

        if len(x) < degree + 1:
            return {"coefficients": [], "r_squared": np.nan, "predictions": []}

        coeffs = np.polyfit(x, y, degree)
        poly = np.poly1d(coeffs)
        y_pred = poly(x)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        return {
            "coefficients": coeffs.tolist(),
            "r_squared": r2,
            "predictions": y_pred.tolist(),
            "x": x.tolist(),
            "y": y.tolist(),
            "poly_fn": poly,
        }

    @staticmethod
    def degradation_by_compound(
        laps: pd.DataFrame, degree: int = 1
    ) -> dict[str, dict]:
        """Fit degradation curves per tire compound."""
        results = {}
        for compound, group in laps.groupby("Compound"):
            results[compound] = LapAnalyzer.fit_degradation_curve(group, degree=degree)
        return results

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    @staticmethod
    def plot_lap_distributions(
        laps: pd.DataFrame,
        group_col: str = "Driver",
        time_col: str = "LapTime",
        kind: str = "violin",
        top_n: int = 10,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Plot lap time distributions as violin or KDE plots.

        Args:
            kind: 'violin', 'kde', or 'box'.
            top_n: Show only the top N groups by median lap time.
        """
        df = laps.copy()
        if hasattr(df[time_col], "dt"):
            df["_seconds"] = df[time_col].dt.total_seconds()
        else:
            df["_seconds"] = df[time_col]

        # Select top N by median
        medians = df.groupby(group_col)["_seconds"].median().nsmallest(top_n)
        df = df[df[group_col].isin(medians.index)]

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        else:
            fig = ax.figure

        if kind == "violin":
            order = medians.index.tolist()
            sns.violinplot(data=df, x=group_col, y="_seconds", order=order, ax=ax)
        elif kind == "kde":
            for name in medians.index:
                subset = df[df[group_col] == name]["_seconds"]
                subset.plot.kde(ax=ax, label=name)
            ax.legend()
        elif kind == "box":
            order = medians.index.tolist()
            sns.boxplot(data=df, x=group_col, y="_seconds", order=order, ax=ax)

        ax.set_ylabel("Lap Time (s)")
        ax.set_title("Lap Time Distribution")
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_degradation(
        laps: pd.DataFrame,
        compound: Optional[str] = None,
        degree: int = 2,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Plot tire degradation curve with fitted polynomial."""
        df = laps.copy()
        if compound:
            df = df[df["Compound"] == compound]

        result = LapAnalyzer.fit_degradation_curve(df, degree=degree)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure

        ax.scatter(result["x"], result["y"], alpha=0.4, s=10, label="Actual")

        if result["coefficients"]:
            x_smooth = np.linspace(min(result["x"]), max(result["x"]), 100)
            y_smooth = result["poly_fn"](x_smooth)
            ax.plot(x_smooth, y_smooth, "r-", linewidth=2, label=f"Fit (R²={result['r_squared']:.3f})")

        ax.set_xlabel("Tire Life (laps)")
        ax.set_ylabel("Lap Time (s)")
        title = f"Tire Degradation — {compound}" if compound else "Tire Degradation"
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        return fig
