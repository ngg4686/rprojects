"""Circuit visualizations.

Generates track maps from GPS position data and overlays speed/gear data.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

logger = logging.getLogger(__name__)


class TrackPlotter:
    """Visualize F1 circuits from telemetry GPS data."""

    @staticmethod
    def plot_track_map(
        telemetry: pd.DataFrame,
        x_col: str = "X",
        y_col: str = "Y",
        color_col: Optional[str] = None,
        cmap: str = "viridis",
        title: str = "Track Map",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Plot a track map from GPS position data, optionally colored by a channel.

        Args:
            telemetry: Telemetry DataFrame with position columns.
            x_col: Column for X coordinate.
            y_col: Column for Y coordinate.
            color_col: Optional column to color the track by (e.g. 'Speed', 'nGear').
            cmap: Matplotlib colormap name.
            title: Plot title.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.figure

        x = telemetry[x_col].values
        y = telemetry[y_col].values

        if color_col and color_col in telemetry.columns:
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            colors = telemetry[color_col].values[:-1]

            norm = mcolors.Normalize(vmin=colors.min(), vmax=colors.max())
            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(colors)
            lc.set_linewidth(3)
            ax.add_collection(lc)
            plt.colorbar(lc, ax=ax, label=color_col)
        else:
            ax.plot(x, y, linewidth=2, color="grey")

        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(False)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_speed_map(
        telemetry: pd.DataFrame,
        title: str = "Speed Map",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Convenience: plot track colored by speed."""
        return TrackPlotter.plot_track_map(
            telemetry, color_col="Speed", cmap="RdYlGn", title=title, ax=ax
        )

    @staticmethod
    def plot_gear_map(
        telemetry: pd.DataFrame,
        title: str = "Gear Map",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Convenience: plot track colored by gear."""
        return TrackPlotter.plot_track_map(
            telemetry, color_col="nGear", cmap="tab10", title=title, ax=ax
        )

    @staticmethod
    def plot_driver_comparison_map(
        telemetry_a: pd.DataFrame,
        telemetry_b: pd.DataFrame,
        driver_a: str,
        driver_b: str,
        channel: str = "Speed",
        ax: Optional[tuple] = None,
    ) -> plt.Figure:
        """Side-by-side track maps comparing two drivers on a channel."""
        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        else:
            axes = ax
            fig = axes[0].figure

        TrackPlotter.plot_track_map(
            telemetry_a,
            color_col=channel,
            title=f"{driver_a} — {channel}",
            ax=axes[0],
        )
        TrackPlotter.plot_track_map(
            telemetry_b,
            color_col=channel,
            title=f"{driver_b} — {channel}",
            ax=axes[1],
        )

        fig.tight_layout()
        return fig
