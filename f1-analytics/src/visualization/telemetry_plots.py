"""Telemetry trace visualization.

Multi-channel telemetry plots for comparing drivers and analyzing
driving patterns across track distance.
"""

import logging
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class TelemetryPlotter:
    """Plot telemetry traces for analysis and comparison."""

    @staticmethod
    def plot_telemetry_channels(
        telemetry: pd.DataFrame,
        channels: Optional[list[str]] = None,
        x_col: str = "Distance",
        title: str = "Telemetry",
        figsize: tuple = (14, 10),
    ) -> plt.Figure:
        """Plot multiple telemetry channels as stacked subplots.

        Args:
            telemetry: Telemetry DataFrame.
            channels: Channels to plot. Defaults to standard set.
            x_col: X-axis column (usually Distance or Time).
        """
        if channels is None:
            channels = ["Speed", "Throttle", "Brake", "nGear", "RPM", "DRS"]
        channels = [c for c in channels if c in telemetry.columns]

        fig, axes = plt.subplots(len(channels), 1, figsize=figsize, sharex=True)
        if len(channels) == 1:
            axes = [axes]

        for ax, ch in zip(axes, channels):
            ax.plot(telemetry[x_col], telemetry[ch], linewidth=0.8)
            ax.set_ylabel(ch)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel(x_col)
        axes[0].set_title(title)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_driver_comparison(
        telemetry_dict: dict[str, pd.DataFrame],
        channel: str = "Speed",
        x_col: str = "Distance",
        title: Optional[str] = None,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Overlay a single telemetry channel for multiple drivers.

        Args:
            telemetry_dict: {driver_name: telemetry_df} mapping.
            channel: Channel to compare.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 5))
        else:
            fig = ax.figure

        for driver, tel in telemetry_dict.items():
            if channel in tel.columns and x_col in tel.columns:
                ax.plot(tel[x_col], tel[channel], label=driver, alpha=0.8, linewidth=0.8)

        ax.set_xlabel(x_col)
        ax.set_ylabel(channel)
        ax.set_title(title or f"{channel} Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_multi_channel_comparison(
        telemetry_dict: dict[str, pd.DataFrame],
        channels: Optional[list[str]] = None,
        x_col: str = "Distance",
        figsize: tuple = (14, 12),
    ) -> plt.Figure:
        """Multi-panel comparison of several channels across drivers."""
        if channels is None:
            channels = ["Speed", "Throttle", "Brake", "nGear"]

        fig, axes = plt.subplots(len(channels), 1, figsize=figsize, sharex=True)
        if len(channels) == 1:
            axes = [axes]

        for ax, ch in zip(axes, channels):
            for driver, tel in telemetry_dict.items():
                if ch in tel.columns and x_col in tel.columns:
                    ax.plot(tel[x_col], tel[ch], label=driver, alpha=0.7, linewidth=0.8)
            ax.set_ylabel(ch)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel(x_col)
        axes[0].set_title("Multi-Channel Driver Comparison")
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_speed_delta(
        tel_a: pd.DataFrame,
        tel_b: pd.DataFrame,
        driver_a: str,
        driver_b: str,
        x_col: str = "Distance",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Plot speed difference between two drivers across track distance.

        Positive values = driver_a is faster.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 4))
        else:
            fig = ax.figure

        # Align by distance via interpolation
        import numpy as np

        d_min = max(tel_a[x_col].min(), tel_b[x_col].min())
        d_max = min(tel_a[x_col].max(), tel_b[x_col].max())
        dist = np.linspace(d_min, d_max, 500)

        speed_a = np.interp(dist, tel_a[x_col].values, tel_a["Speed"].values)
        speed_b = np.interp(dist, tel_b[x_col].values, tel_b["Speed"].values)
        delta = speed_a - speed_b

        ax.fill_between(dist, delta, where=(delta > 0), color="green", alpha=0.5, label=f"{driver_a} faster")
        ax.fill_between(dist, delta, where=(delta < 0), color="red", alpha=0.5, label=f"{driver_b} faster")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Speed Delta (km/h)")
        ax.set_title(f"Speed Delta: {driver_a} vs {driver_b}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
