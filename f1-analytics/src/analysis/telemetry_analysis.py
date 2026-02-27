"""Telemetry pattern analysis.

Clusters drivers by driving style using telemetry features, DTW, and
dimensionality reduction (PCA / t-SNE).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class TelemetryAnalyzer:
    """Analyze and cluster telemetry traces."""

    # ------------------------------------------------------------------
    # Feature extraction from telemetry
    # ------------------------------------------------------------------

    @staticmethod
    def extract_telemetry_features(telemetry: pd.DataFrame) -> dict:
        """Extract summary features from a single-lap telemetry trace.

        Features include braking statistics, throttle application patterns,
        speed statistics, and gear usage.
        """
        features = {}

        if "Speed" in telemetry.columns:
            speed = telemetry["Speed"]
            features["speed_mean"] = speed.mean()
            features["speed_std"] = speed.std()
            features["speed_max"] = speed.max()
            features["speed_min"] = speed.min()

        if "Throttle" in telemetry.columns:
            throttle = telemetry["Throttle"]
            features["throttle_mean"] = throttle.mean()
            features["full_throttle_pct"] = (throttle >= 98).mean()
            features["off_throttle_pct"] = (throttle <= 2).mean()
            features["throttle_std"] = throttle.std()

        if "Brake" in telemetry.columns:
            brake = telemetry["Brake"]
            features["brake_pct"] = (brake > 0).mean()
            braking_zones = (brake > 0).astype(int).diff().fillna(0)
            features["num_braking_zones"] = (braking_zones == 1).sum()

        if "nGear" in telemetry.columns:
            gear = telemetry["nGear"]
            features["gear_changes"] = (gear.diff().fillna(0) != 0).sum()
            features["avg_gear"] = gear.mean()

        if "DRS" in telemetry.columns:
            features["drs_pct"] = (telemetry["DRS"] > 0).mean()

        return features

    @staticmethod
    def extract_features_for_session(
        laps: pd.DataFrame, drivers: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """Extract telemetry features for all laps in a session.

        Args:
            laps: FastF1 laps object with get_telemetry() available.
            drivers: Optional list of driver abbreviations to include.

        Returns:
            DataFrame with one row per lap, columns are features.
        """
        rows = []
        if drivers:
            laps = laps[laps["Driver"].isin(drivers)]

        for idx, lap in laps.iterrows():
            try:
                tel = lap.get_telemetry()
                feats = TelemetryAnalyzer.extract_telemetry_features(tel)
                feats["Driver"] = lap["Driver"]
                feats["LapNumber"] = lap["LapNumber"]
                if "Compound" in lap.index:
                    feats["Compound"] = lap["Compound"]
                rows.append(feats)
            except Exception:
                continue

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # DTW distance
    # ------------------------------------------------------------------

    @staticmethod
    def compute_dtw_distance(
        trace_a: np.ndarray, trace_b: np.ndarray
    ) -> float:
        """Compute DTW distance between two 1D telemetry traces."""
        try:
            from dtw import dtw as dtw_func
            alignment = dtw_func(trace_a, trace_b)
            return alignment.distance
        except ImportError:
            logger.warning("dtw-python not installed, falling back to euclidean distance")
            min_len = min(len(trace_a), len(trace_b))
            return np.sqrt(np.sum((trace_a[:min_len] - trace_b[:min_len]) ** 2))

    @staticmethod
    def compute_dtw_matrix(traces: list[np.ndarray]) -> np.ndarray:
        """Compute a pairwise DTW distance matrix for a list of traces.

        Args:
            traces: List of 1D arrays (e.g. speed traces per lap).

        Returns:
            Symmetric distance matrix of shape (n, n).
        """
        n = len(traces)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                d = TelemetryAnalyzer.compute_dtw_distance(traces[i], traces[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        return dist_matrix

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    @staticmethod
    def cluster_drivers(
        feature_df: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
        n_clusters: int = 4,
        method: str = "kmeans",
    ) -> pd.DataFrame:
        """Cluster drivers by their aggregated telemetry features.

        Args:
            feature_df: DataFrame with driver features (one row per driver).
            feature_cols: Columns to use. If None, all numeric columns.
            n_clusters: Number of clusters.
            method: 'kmeans' or 'agglomerative'.

        Returns:
            Input DataFrame with 'cluster' column added.
        """
        if feature_cols is None:
            feature_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()

        X = feature_df[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if method == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        else:
            model = AgglomerativeClustering(n_clusters=n_clusters)

        feature_df = feature_df.copy()
        feature_df["cluster"] = model.fit_predict(X_scaled)
        return feature_df

    # ------------------------------------------------------------------
    # Dimensionality reduction
    # ------------------------------------------------------------------

    @staticmethod
    def reduce_dimensions(
        feature_df: pd.DataFrame,
        feature_cols: Optional[list[str]] = None,
        method: str = "pca",
        n_components: int = 2,
    ) -> pd.DataFrame:
        """Reduce feature space to 2D for visualization.

        Args:
            method: 'pca' or 'tsne'.
        """
        if feature_cols is None:
            feature_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()

        X = feature_df[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=42)
        else:
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(X) - 1))

        coords = reducer.fit_transform(X_scaled)

        result = feature_df.copy()
        result["dim_1"] = coords[:, 0]
        result["dim_2"] = coords[:, 1]
        return result

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    @staticmethod
    def plot_driver_clusters(
        feature_df: pd.DataFrame,
        method: str = "pca",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Scatter plot of drivers in reduced 2D feature space, colored by cluster."""
        df = feature_df.copy()
        if "dim_1" not in df.columns:
            df = TelemetryAnalyzer.reduce_dimensions(df, method=method)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure

        color_col = "cluster" if "cluster" in df.columns else "Driver"
        scatter = ax.scatter(
            df["dim_1"], df["dim_2"],
            c=df[color_col].astype("category").cat.codes if df[color_col].dtype == "object" else df[color_col],
            cmap="tab10", alpha=0.7, s=60,
        )

        if "Driver" in df.columns:
            for _, row in df.iterrows():
                ax.annotate(row["Driver"], (row["dim_1"], row["dim_2"]),
                            fontsize=8, alpha=0.8)

        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        ax.set_title(f"Driver Telemetry Clusters ({method.upper()})")
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_braking_profiles(
        telemetry_dict: dict[str, pd.DataFrame],
        distance_col: str = "Distance",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Figure:
        """Overlay braking traces for multiple drivers.

        Args:
            telemetry_dict: {driver_name: telemetry_df} mapping.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 5))
        else:
            fig = ax.figure

        for driver, tel in telemetry_dict.items():
            if "Brake" in tel.columns and distance_col in tel.columns:
                ax.plot(tel[distance_col], tel["Brake"], label=driver, alpha=0.7)

        ax.set_xlabel("Track Distance (m)")
        ax.set_ylabel("Brake Pressure")
        ax.set_title("Braking Profiles Comparison")
        ax.legend()
        fig.tight_layout()
        return fig
