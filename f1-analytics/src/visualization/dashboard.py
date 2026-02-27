"""Summary dashboard generation.

Creates multi-panel summary figures combining track maps, telemetry,
lap distributions, and key statistics for a session or race weekend.
"""

import logging
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .track_plots import TrackPlotter
from .telemetry_plots import TelemetryPlotter

logger = logging.getLogger(__name__)


class DashboardGenerator:
    """Generate summary dashboard figures."""

    @staticmethod
    def session_dashboard(
        laps: pd.DataFrame,
        fastest_telemetry: pd.DataFrame,
        weather: Optional[pd.DataFrame] = None,
        title: str = "Session Dashboard",
    ) -> plt.Figure:
        """Create a multi-panel session overview dashboard.

        Panels:
        1. Track map (speed-colored from fastest lap)
        2. Lap time distribution (violin)
        3. Fastest lap telemetry channels
        4. Tire compound usage / session info
        """
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Panel 1: Track map
        ax1 = fig.add_subplot(gs[0, 0])
        if "X" in fastest_telemetry.columns and "Y" in fastest_telemetry.columns:
            TrackPlotter.plot_speed_map(fastest_telemetry, title="Fastest Lap — Speed", ax=ax1)
        else:
            ax1.text(0.5, 0.5, "No GPS data available", ha="center", va="center")
            ax1.set_title("Track Map")

        # Panel 2: Lap time distribution
        ax2 = fig.add_subplot(gs[0, 1])
        if "LapTime" in laps.columns and "Driver" in laps.columns:
            from ..analysis.lap_analysis import LapAnalyzer
            filtered = LapAnalyzer.filter_representative_laps(laps)
            LapAnalyzer.plot_lap_distributions(filtered, ax=ax2, top_n=10)

        # Panel 3: Telemetry channels (bottom-left, spanning full width)
        ax3 = fig.add_subplot(gs[1, :])
        channels = ["Speed", "Throttle", "Brake"]
        available = [c for c in channels if c in fastest_telemetry.columns]
        x_col = "Distance" if "Distance" in fastest_telemetry.columns else fastest_telemetry.index.name or "index"

        for ch in available:
            if x_col in fastest_telemetry.columns:
                ax3.plot(fastest_telemetry[x_col], fastest_telemetry[ch], label=ch, alpha=0.8)
        ax3.set_title("Fastest Lap Telemetry")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Panel 4: Session stats
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.axis("off")
        stats_text = _build_session_stats(laps, weather)
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                 fontsize=10, verticalalignment="top", fontfamily="monospace")
        ax4.set_title("Session Statistics")

        # Panel 5: Compound distribution
        ax5 = fig.add_subplot(gs[2, 1])
        if "Compound" in laps.columns:
            compound_counts = laps["Compound"].value_counts()
            colors = _compound_colors(compound_counts.index)
            compound_counts.plot.bar(ax=ax5, color=colors)
            ax5.set_title("Tire Compound Usage")
            ax5.set_ylabel("Number of Laps")

        fig.suptitle(title, fontsize=16, y=1.0)
        return fig

    @staticmethod
    def race_summary(
        results: pd.DataFrame,
        laps: pd.DataFrame,
        pit_stops: Optional[pd.DataFrame] = None,
        title: str = "Race Summary",
    ) -> plt.Figure:
        """Race result and strategy summary dashboard."""
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

        # Finishing order
        ax1 = fig.add_subplot(gs[0, 0])
        if "position" in results.columns and "driver_code" in results.columns:
            top10 = results.nsmallest(10, "position")
            ax1.barh(top10["driver_code"], top10["points"])
            ax1.invert_yaxis()
            ax1.set_xlabel("Points")
            ax1.set_title("Top 10 Finishers")

        # Grid vs finish
        ax2 = fig.add_subplot(gs[0, 1])
        if "grid" in results.columns and "position" in results.columns:
            from ..analysis.race_analysis import RaceAnalyzer
            RaceAnalyzer.plot_grid_vs_finish(results, ax=ax2)

        # Lap time evolution
        ax3 = fig.add_subplot(gs[1, :])
        if "LapTime" in laps.columns and "LapNumber" in laps.columns:
            for driver in laps["Driver"].unique()[:6]:
                dlaps = laps[laps["Driver"] == driver]
                if hasattr(dlaps["LapTime"], "dt"):
                    times = dlaps["LapTime"].dt.total_seconds()
                else:
                    times = dlaps["LapTime"]
                ax3.plot(dlaps["LapNumber"], times, label=driver, alpha=0.7, linewidth=0.8)
            ax3.set_xlabel("Lap Number")
            ax3.set_ylabel("Lap Time (s)")
            ax3.set_title("Lap Time Evolution")
            ax3.legend(loc="upper right")
            ax3.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=16)
        return fig


def _build_session_stats(laps: pd.DataFrame, weather: Optional[pd.DataFrame]) -> str:
    """Build a text block of session statistics."""
    lines = []
    if "Driver" in laps.columns:
        lines.append(f"Drivers: {laps['Driver'].nunique()}")
    lines.append(f"Total laps: {len(laps)}")

    if "LapTime" in laps.columns:
        if hasattr(laps["LapTime"], "dt"):
            fastest = laps["LapTime"].dt.total_seconds().min()
        else:
            fastest = laps["LapTime"].min()
        lines.append(f"Fastest lap: {fastest:.3f}s")

    if weather is not None and not weather.empty:
        if "AirTemp" in weather.columns:
            lines.append(f"Air temp: {weather['AirTemp'].mean():.1f}°C")
        if "TrackTemp" in weather.columns:
            lines.append(f"Track temp: {weather['TrackTemp'].mean():.1f}°C")
        if "Rainfall" in weather.columns:
            lines.append(f"Rain: {'Yes' if weather['Rainfall'].any() else 'No'}")

    return "\n".join(lines)


def _compound_colors(compounds: pd.Index) -> list[str]:
    """Map tire compound names to F1-style colors."""
    color_map = {
        "SOFT": "#FF3333",
        "MEDIUM": "#FFD700",
        "HARD": "#FFFFFF",
        "INTERMEDIATE": "#39B54A",
        "WET": "#0072CE",
    }
    return [color_map.get(c.upper(), "#888888") for c in compounds]
