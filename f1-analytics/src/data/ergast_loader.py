"""Jolpica-F1 API client (Ergast successor) for historical race data.

Provides access to the Jolpica-F1 REST API, which mirrors the old Ergast
schema and covers results, standings, lap times, and pit stops back to 1950.
"""

import json
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

from config import JOLPICA_BASE_URL, RAW_DATA_DIR

logger = logging.getLogger(__name__)

CACHE_DIR = RAW_DATA_DIR / "jolpica_cache"


class JolpicaClient:
    """REST client for the Jolpica-F1 (Ergast) API with local caching."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        rate_limit_pause: float = 0.25,
    ):
        self.base_url = (base_url or JOLPICA_BASE_URL).rstrip("/")
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_pause = rate_limit_pause
        self._session = requests.Session()

    def _cache_path(self, url: str) -> Path:
        key = hashlib.sha256(url.encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _fetch_json(self, path: str, use_cache: bool = True) -> dict:
        """Fetch a JSON response from the Jolpica API.

        The API uses .json suffix on endpoints for JSON responses.
        Handles pagination via limit/offset.
        """
        url = f"{self.base_url}/{path}.json"
        cache = self._cache_path(url)

        if use_cache and cache.exists():
            return json.loads(cache.read_text())

        logger.info("Fetching %s", url)
        resp = self._session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if use_cache:
            cache.write_text(json.dumps(data))

        time.sleep(self.rate_limit_pause)
        return data

    def _extract_table(self, data: dict, table_key: str) -> list[dict]:
        """Navigate the Ergast-style nested JSON to extract a result table."""
        mrdata = data.get("MRData", data)
        for key, val in mrdata.items():
            if isinstance(val, dict):
                if table_key in val:
                    return val[table_key]
        return []

    # ------------------------------------------------------------------
    # Race results
    # ------------------------------------------------------------------

    def get_race_results(self, year: int, round_num: Optional[int] = None) -> pd.DataFrame:
        """Get race results for a season or a specific round."""
        path = f"{year}/results" if round_num is None else f"{year}/{round_num}/results"
        data = self._fetch_json(path)
        races = self._extract_table(data, "Races")

        rows = []
        for race in races:
            for result in race.get("Results", []):
                rows.append(
                    {
                        "season": year,
                        "round": int(race["round"]),
                        "raceName": race["raceName"],
                        "circuitId": race["Circuit"]["circuitId"],
                        "date": race["date"],
                        "driverId": result["Driver"]["driverId"],
                        "driver_code": result["Driver"].get("code", ""),
                        "constructorId": result["Constructor"]["constructorId"],
                        "grid": int(result["grid"]),
                        "position": int(result["position"]) if result["position"].isdigit() else None,
                        "points": float(result["points"]),
                        "status": result["status"],
                        "laps_completed": int(result["laps"]),
                    }
                )
        return pd.DataFrame(rows)

    def get_qualifying(self, year: int, round_num: Optional[int] = None) -> pd.DataFrame:
        """Get qualifying results."""
        path = f"{year}/qualifying" if round_num is None else f"{year}/{round_num}/qualifying"
        data = self._fetch_json(path)
        races = self._extract_table(data, "Races")

        rows = []
        for race in races:
            for result in race.get("QualifyingResults", []):
                rows.append(
                    {
                        "season": year,
                        "round": int(race["round"]),
                        "raceName": race["raceName"],
                        "driverId": result["Driver"]["driverId"],
                        "driver_code": result["Driver"].get("code", ""),
                        "constructorId": result["Constructor"]["constructorId"],
                        "position": int(result["position"]),
                        "Q1": result.get("Q1"),
                        "Q2": result.get("Q2"),
                        "Q3": result.get("Q3"),
                    }
                )
        return pd.DataFrame(rows)

    def get_driver_standings(self, year: int) -> pd.DataFrame:
        """Get end-of-season (or current) driver standings."""
        data = self._fetch_json(f"{year}/driverStandings")
        standings_lists = self._extract_table(data, "StandingsLists")

        rows = []
        for sl in standings_lists:
            for s in sl.get("DriverStandings", []):
                rows.append(
                    {
                        "season": year,
                        "position": int(s["position"]),
                        "points": float(s["points"]),
                        "wins": int(s["wins"]),
                        "driverId": s["Driver"]["driverId"],
                        "driver_code": s["Driver"].get("code", ""),
                        "constructorId": s["Constructors"][0]["constructorId"]
                        if s.get("Constructors")
                        else None,
                    }
                )
        return pd.DataFrame(rows)

    def get_constructor_standings(self, year: int) -> pd.DataFrame:
        """Get end-of-season (or current) constructor standings."""
        data = self._fetch_json(f"{year}/constructorStandings")
        standings_lists = self._extract_table(data, "StandingsLists")

        rows = []
        for sl in standings_lists:
            for s in sl.get("ConstructorStandings", []):
                rows.append(
                    {
                        "season": year,
                        "position": int(s["position"]),
                        "points": float(s["points"]),
                        "wins": int(s["wins"]),
                        "constructorId": s["Constructor"]["constructorId"],
                    }
                )
        return pd.DataFrame(rows)

    def get_lap_times(self, year: int, round_num: int) -> pd.DataFrame:
        """Get lap-by-lap times for a specific race."""
        all_laps: list[dict] = []
        lap_num = 1

        while True:
            data = self._fetch_json(f"{year}/{round_num}/laps/{lap_num}")
            races = self._extract_table(data, "Races")
            if not races or not races[0].get("Laps"):
                break

            for lap in races[0]["Laps"]:
                for timing in lap.get("Timings", []):
                    all_laps.append(
                        {
                            "season": year,
                            "round": round_num,
                            "lap": int(lap["number"]),
                            "driverId": timing["driverId"],
                            "position": int(timing.get("position", 0)),
                            "time": timing.get("time"),
                        }
                    )
            lap_num += 1

        return pd.DataFrame(all_laps)

    def get_pit_stops(self, year: int, round_num: int) -> pd.DataFrame:
        """Get pit stop data for a specific race."""
        data = self._fetch_json(f"{year}/{round_num}/pitstops")
        races = self._extract_table(data, "Races")

        rows = []
        for race in races:
            for pit in race.get("PitStops", []):
                rows.append(
                    {
                        "season": year,
                        "round": round_num,
                        "driverId": pit["driverId"],
                        "stop": int(pit["stop"]),
                        "lap": int(pit["lap"]),
                        "time": pit.get("time"),
                        "duration": pit.get("duration"),
                    }
                )
        return pd.DataFrame(rows)

    def get_circuits(self) -> pd.DataFrame:
        """Get the list of all circuits."""
        data = self._fetch_json("circuits")
        circuits = self._extract_table(data, "Circuits")
        return pd.DataFrame(
            [
                {
                    "circuitId": c["circuitId"],
                    "name": c["circuitName"],
                    "lat": float(c["Location"]["lat"]),
                    "lng": float(c["Location"]["long"]),
                    "locality": c["Location"]["locality"],
                    "country": c["Location"]["country"],
                }
                for c in circuits
            ]
        )

    def get_all_results(self, start_year: int = 2018, end_year: int = 2024) -> pd.DataFrame:
        """Download race results for a range of seasons."""
        frames = []
        for year in range(start_year, end_year + 1):
            try:
                df = self.get_race_results(year)
                frames.append(df)
                logger.info("Loaded results for %d (%d rows)", year, len(df))
            except Exception as e:
                logger.warning("Failed to load %d results: %s", year, e)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
