"""OpenF1 API client for supplementary telemetry and event data.

Provides access to the OpenF1 REST API (https://api.openf1.org/v1/)
with built-in request caching and rate-limit handling.
"""

import json
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

from config import OPENF1_BASE_URL, RAW_DATA_DIR

logger = logging.getLogger(__name__)

CACHE_DIR = RAW_DATA_DIR / "openf1_cache"


class OpenF1Client:
    """Simple REST client for the OpenF1 API with local disk caching."""

    ENDPOINTS = [
        "car_data",
        "drivers",
        "intervals",
        "laps",
        "location",
        "meetings",
        "pit",
        "position",
        "race_control",
        "sessions",
        "stints",
        "team_radio",
        "weather",
    ]

    def __init__(
        self,
        base_url: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        rate_limit_pause: float = 0.5,
    ):
        self.base_url = (base_url or OPENF1_BASE_URL).rstrip("/")
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_pause = rate_limit_pause
        self._session = requests.Session()

    def _cache_key(self, endpoint: str, params: dict) -> str:
        raw = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[list[dict]]:
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def _save_cache(self, key: str, data: list[dict]) -> None:
        path = self.cache_dir / f"{key}.json"
        path.write_text(json.dumps(data))

    def fetch(
        self, endpoint: str, use_cache: bool = True, **params: Any
    ) -> list[dict]:
        """Fetch data from an OpenF1 endpoint.

        Args:
            endpoint: API endpoint name (e.g. 'laps', 'car_data').
            use_cache: Whether to use local disk cache.
            **params: Query parameters forwarded to the API.

        Returns:
            List of result dicts.
        """
        params = {k: v for k, v in params.items() if v is not None}
        cache_key = self._cache_key(endpoint, params)

        if use_cache:
            cached = self._get_cached(cache_key)
            if cached is not None:
                logger.debug("Cache hit for %s %s", endpoint, params)
                return cached

        url = f"{self.base_url}/{endpoint}"
        logger.info("Fetching %s with %s", url, params)

        response = self._session.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if use_cache:
            self._save_cache(cache_key, data)

        time.sleep(self.rate_limit_pause)
        return data

    def fetch_df(self, endpoint: str, **params: Any) -> pd.DataFrame:
        """Fetch data and return as a DataFrame."""
        data = self.fetch(endpoint, **params)
        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def get_sessions(self, year: int) -> pd.DataFrame:
        """List all sessions for a season."""
        return self.fetch_df("sessions", year=year)

    def get_laps(
        self,
        session_key: int,
        driver_number: Optional[int] = None,
    ) -> pd.DataFrame:
        return self.fetch_df(
            "laps", session_key=session_key, driver_number=driver_number
        )

    def get_car_data(
        self,
        session_key: int,
        driver_number: int,
    ) -> pd.DataFrame:
        """Fetch high-frequency car telemetry (~3.7 Hz)."""
        return self.fetch_df(
            "car_data",
            session_key=session_key,
            driver_number=driver_number,
        )

    def get_position(self, session_key: int) -> pd.DataFrame:
        return self.fetch_df("position", session_key=session_key)

    def get_pit_stops(self, session_key: int) -> pd.DataFrame:
        return self.fetch_df("pit", session_key=session_key)

    def get_weather(self, session_key: int) -> pd.DataFrame:
        return self.fetch_df("weather", session_key=session_key)

    def get_race_control(self, session_key: int) -> pd.DataFrame:
        return self.fetch_df("race_control", session_key=session_key)

    def get_team_radio(
        self, session_key: int, driver_number: Optional[int] = None
    ) -> pd.DataFrame:
        return self.fetch_df(
            "team_radio",
            session_key=session_key,
            driver_number=driver_number,
        )
