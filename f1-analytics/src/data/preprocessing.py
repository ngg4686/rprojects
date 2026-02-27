"""Data cleaning, feature engineering, and dataset preparation.

Handles merging telemetry with lap metadata, normalizing traces to track
distance, temporal train/val/test splitting, and sliding-window creation
for sequence models.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR, TELEMETRY_CHANNELS

logger = logging.getLogger(__name__)


class F1Preprocessor:
    """End-to-end preprocessing pipeline for F1 data."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or PROCESSED_DATA_DIR

    # ------------------------------------------------------------------
    # Telemetry enrichment
    # ------------------------------------------------------------------

    @staticmethod
    def merge_telemetry_with_laps(
        telemetry: pd.DataFrame, laps: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge high-frequency telemetry with per-lap metadata.

        Adds compound, lap number, stint, and estimated fuel load to each
        telemetry sample based on the lap it belongs to.
        """
        merged = telemetry.copy()

        lap_cols = ["LapNumber", "Compound", "Stint", "TyreLife", "LapTime", "Driver"]
        available = [c for c in lap_cols if c in laps.columns]

        if "LapNumber" in telemetry.columns and "LapNumber" in laps.columns:
            merged = merged.merge(
                laps[available].drop_duplicates(subset=["LapNumber"]),
                on="LapNumber",
                how="left",
                suffixes=("", "_lap"),
            )

        # Estimate fuel load (linear approximation: full tank minus ~1.6 kg/lap)
        if "LapNumber" in merged.columns:
            max_laps = merged["LapNumber"].max()
            merged["FuelLoadEstimate"] = 1.0 - (merged["LapNumber"] / max_laps)

        return merged

    # ------------------------------------------------------------------
    # Distance normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_to_distance(
        telemetry: pd.DataFrame,
        distance_col: str = "Distance",
        num_points: int = 500,
    ) -> pd.DataFrame:
        """Resample telemetry to evenly spaced distance points.

        This enables cross-driver comparison by aligning traces to track
        distance rather than time.

        Args:
            telemetry: Raw telemetry DataFrame.
            distance_col: Column containing distance-along-track values.
            num_points: Number of evenly spaced output samples.

        Returns:
            Resampled DataFrame indexed by normalized distance.
        """
        if distance_col not in telemetry.columns:
            logger.warning("Distance column '%s' not found", distance_col)
            return telemetry

        df = telemetry.sort_values(distance_col).copy()
        d_min, d_max = df[distance_col].min(), df[distance_col].max()

        if d_max <= d_min:
            return df

        new_dist = np.linspace(d_min, d_max, num_points)
        resampled = {distance_col: new_dist}

        for col in df.columns:
            if col == distance_col:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                resampled[col] = np.interp(new_dist, df[distance_col].values, df[col].values)

        return pd.DataFrame(resampled)

    # ------------------------------------------------------------------
    # Missing data handling
    # ------------------------------------------------------------------

    @staticmethod
    def interpolate_missing(
        telemetry: pd.DataFrame,
        channels: Optional[list[str]] = None,
        method: str = "linear",
        max_gap: int = 10,
    ) -> pd.DataFrame:
        """Fill missing or corrupt telemetry samples via interpolation.

        Args:
            telemetry: Telemetry DataFrame.
            channels: Columns to interpolate. Defaults to standard channels.
            method: Interpolation method (passed to pandas).
            max_gap: Maximum consecutive NaNs to fill.
        """
        channels = channels or TELEMETRY_CHANNELS
        df = telemetry.copy()

        for ch in channels:
            if ch in df.columns:
                df[ch] = df[ch].interpolate(method=method, limit=max_gap)

        return df

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    @staticmethod
    def engineer_features(telemetry: pd.DataFrame) -> pd.DataFrame:
        """Derive higher-level features from raw telemetry.

        Adds: sector deltas, braking points, apex speeds, acceleration zones.
        """
        df = telemetry.copy()

        # Acceleration (speed derivative with respect to distance or index)
        if "Speed" in df.columns:
            df["Acceleration"] = df["Speed"].diff().fillna(0)

        # Braking indicator: Brake > 0 or large negative acceleration
        if "Brake" in df.columns:
            df["IsBraking"] = (df["Brake"] > 0).astype(int)
        elif "Acceleration" in df.columns:
            df["IsBraking"] = (df["Acceleration"] < -2).astype(int)

        # Full throttle indicator
        if "Throttle" in df.columns:
            df["IsFullThrottle"] = (df["Throttle"] >= 98).astype(int)

        # Coasting indicator (neither braking nor full throttle)
        if "IsBraking" in df.columns and "IsFullThrottle" in df.columns:
            df["IsCoasting"] = ((df["IsBraking"] == 0) & (df["IsFullThrottle"] == 0)).astype(int)

        # Braking points: transitions from not braking to braking
        if "IsBraking" in df.columns:
            df["BrakingPoint"] = (df["IsBraking"].diff() == 1).astype(int)

        return df

    @staticmethod
    def compute_sector_deltas(
        laps: pd.DataFrame, reference_driver: str
    ) -> pd.DataFrame:
        """Compute sector time deltas relative to a reference driver.

        Args:
            laps: Laps DataFrame containing Sector1Time, Sector2Time, etc.
            reference_driver: Driver abbreviation to use as baseline.

        Returns:
            DataFrame with delta columns added.
        """
        sector_cols = [c for c in laps.columns if c.startswith("Sector") and c.endswith("Time")]
        if not sector_cols:
            return laps

        df = laps.copy()
        ref = df[df["Driver"] == reference_driver]

        for col in sector_cols:
            ref_median = ref[col].dt.total_seconds().median() if hasattr(ref[col], "dt") else ref[col].median()
            if hasattr(df[col], "dt"):
                df[f"{col}Delta"] = df[col].dt.total_seconds() - ref_median
            else:
                df[f"{col}Delta"] = df[col] - ref_median

        return df

    # ------------------------------------------------------------------
    # Sliding windows for sequence models
    # ------------------------------------------------------------------

    @staticmethod
    def create_sliding_windows(
        telemetry: pd.DataFrame,
        channels: Optional[list[str]] = None,
        window_size: int = 200,
        stride: int = 50,
    ) -> np.ndarray:
        """Create fixed-length sliding windows from telemetry.

        Args:
            telemetry: Telemetry DataFrame.
            channels: Columns to include. Defaults to standard channels.
            window_size: Length of each window (number of samples).
            stride: Step between consecutive windows.

        Returns:
            Array of shape (num_windows, window_size, num_channels).
        """
        channels = channels or TELEMETRY_CHANNELS
        available = [c for c in channels if c in telemetry.columns]
        data = telemetry[available].values

        windows = []
        for start in range(0, len(data) - window_size + 1, stride):
            windows.append(data[start : start + window_size])

        if not windows:
            return np.empty((0, window_size, len(available)))
        return np.stack(windows)

    # ------------------------------------------------------------------
    # Train / val / test split
    # ------------------------------------------------------------------

    @staticmethod
    def temporal_split(
        df: pd.DataFrame,
        date_col: str = "date",
        val_frac: float = 0.15,
        test_frac: float = 0.15,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data by date to avoid future data leakage.

        Args:
            df: DataFrame with a date column.
            date_col: Column name containing dates.
            val_frac: Fraction of data for validation.
            test_frac: Fraction of data for testing.

        Returns:
            (train, val, test) DataFrames.
        """
        df = df.sort_values(date_col).reset_index(drop=True)
        n = len(df)
        train_end = int(n * (1 - val_frac - test_frac))
        val_end = int(n * (1 - test_frac))

        return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

    # ------------------------------------------------------------------
    # Save / load helpers
    # ------------------------------------------------------------------

    def save_processed(self, df: pd.DataFrame, name: str) -> None:
        """Save a processed DataFrame as parquet."""
        path = self.output_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Saved %s (%d rows) to %s", name, len(df), path)

    def load_processed(self, name: str) -> pd.DataFrame:
        """Load a processed parquet file."""
        path = self.output_dir / f"{name}.parquet"
        return pd.read_parquet(path)
