"""Project-wide configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent

# Data directories
FASTF1_CACHE_DIR = PROJECT_ROOT / os.getenv("FASTF1_CACHE_DIR", "data/cache")
RAW_DATA_DIR = PROJECT_ROOT / os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")

# API endpoints
OPENF1_BASE_URL = os.getenv("OPENF1_BASE_URL", "https://api.openf1.org/v1")
JOLPICA_BASE_URL = os.getenv("JOLPICA_BASE_URL", "https://api.jolpi.ca/ergast/f1")

# Model checkpoints
CHECKPOINT_DIR = PROJECT_ROOT / os.getenv("CHECKPOINT_DIR", "models/checkpoints")

# Training defaults
DEFAULT_DEVICE = os.getenv("DEFAULT_DEVICE", "cpu")
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", "32"))
DEFAULT_LEARNING_RATE = float(os.getenv("DEFAULT_LEARNING_RATE", "0.001"))
DEFAULT_EPOCHS = int(os.getenv("DEFAULT_EPOCHS", "50"))

# Seasons available for analysis
AVAILABLE_SEASONS = list(range(2018, 2026))
DEFAULT_SEASONS = [2023, 2024]

# Telemetry channels used across the project
TELEMETRY_CHANNELS = ["Speed", "Throttle", "Brake", "nGear", "RPM", "DRS"]
TELEMETRY_CHANNELS_NORMALIZED = ["speed", "throttle", "brake", "gear", "rpm", "drs"]

# Ensure directories exist
for d in [FASTF1_CACHE_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
