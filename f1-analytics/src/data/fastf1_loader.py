"""FastF1 telemetry data acquisition module.

Wraps the FastF1 library to provide cached access to F1 session data,
lap times, telemetry traces, and weather information.
"""

import logging
from typing import Optional

import fastf1
import pandas as pd

from config import FASTF1_CACHE_DIR, AVAILABLE_SEASONS

logger = logging.getLogger(__name__)


class FastF1Loader:
    """Load and cache F1 session data via the FastF1 library."""

    def __init__(self, cache_dir: Optional[str] = None):
        cache_path = str(cache_dir or FASTF1_CACHE_DIR)
        fastf1.Cache.enable_cache(cache_path)
        logger.info("FastF1 cache enabled at %s", cache_path)

    def get_session(
        self, year: int, gp: str | int, identifier: str = "R"
    ) -> fastf1.core.Session:
        """Load a full session (race, qualifying, practice).

        Args:
            year: Season year.
            gp: Grand Prix name or round number.
            identifier: Session type — 'R' (race), 'Q' (qualifying),
                        'FP1', 'FP2', 'FP3', 'S' (sprint).

        Returns:
            Loaded FastF1 Session object.
        """
        session = fastf1.get_session(year, gp, identifier)
        session.load()
        logger.info("Loaded %s %s %s", year, gp, identifier)
        return session

    def get_laps(
        self, year: int, gp: str | int, identifier: str = "R"
    ) -> pd.DataFrame:
        """Return the laps DataFrame for a session."""
        session = self.get_session(year, gp, identifier)
        return session.laps

    def get_driver_telemetry(
        self,
        year: int,
        gp: str | int,
        driver: str,
        identifier: str = "R",
        lap_number: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return telemetry for a specific driver and optionally a single lap.

        Args:
            year: Season year.
            gp: Grand Prix name or round number.
            driver: Three-letter driver abbreviation (e.g. 'VER').
            identifier: Session type.
            lap_number: If provided, return telemetry for this lap only.

        Returns:
            Telemetry DataFrame with columns like Speed, Throttle, Brake, etc.
        """
        session = self.get_session(year, gp, identifier)
        driver_laps = session.laps.pick_driver(driver)

        if lap_number is not None:
            driver_laps = driver_laps[driver_laps["LapNumber"] == lap_number]

        telemetry = driver_laps.get_telemetry()
        return telemetry

    def get_fastest_lap_telemetry(
        self, year: int, gp: str | int, driver: str, identifier: str = "R"
    ) -> pd.DataFrame:
        """Return telemetry for a driver's fastest lap in a session."""
        session = self.get_session(year, gp, identifier)
        fastest = session.laps.pick_driver(driver).pick_fastest()
        return fastest.get_telemetry()

    def get_event_schedule(self, year: int) -> pd.DataFrame:
        """Return the event schedule for a given season."""
        return fastf1.get_event_schedule(year)

    def get_all_race_laps(
        self, seasons: Optional[list[int]] = None
    ) -> pd.DataFrame:
        """Download laps for all races across multiple seasons.

        This can be slow on first run. Results are cached by FastF1.

        Args:
            seasons: List of season years. Defaults to DEFAULT_SEASONS.

        Returns:
            Combined DataFrame of all race laps.
        """
        seasons = seasons or [2023, 2024]
        all_laps = []

        for year in seasons:
            schedule = self.get_event_schedule(year)
            for _, event in schedule.iterrows():
                if event["EventFormat"] == "testing":
                    continue
                try:
                    session = self.get_session(year, event["EventName"], "R")
                    laps = session.laps.copy()
                    laps["Season"] = year
                    laps["EventName"] = event["EventName"]
                    all_laps.append(laps)
                    logger.info("Loaded %s %s", year, event["EventName"])
                except Exception as e:
                    logger.warning(
                        "Failed to load %s %s: %s",
                        year,
                        event["EventName"],
                        e,
                    )

        if not all_laps:
            return pd.DataFrame()
        return pd.concat(all_laps, ignore_index=True)

    def get_weather(
        self, year: int, gp: str | int, identifier: str = "R"
    ) -> pd.DataFrame:
        """Return weather data for a session."""
        session = self.get_session(year, gp, identifier)
        return session.weather_data
