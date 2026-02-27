# F1 Analytics — Architecture

## Project Goal

Hypothesis-driven F1 prediction system. Pulls from multiple data sources, builds testable hypotheses about race outcomes, generates pre-race predictions, scores them against reality, and iteratively improves. Combines statistical analysis and deep learning with a concrete feedback loop.

## Directory Structure

```
f1-analytics/
├── README.md
├── ARCHITECTURE.md          # This file
├── requirements.txt
├── .env                     # Environment variables (not committed)
├── .gitignore
├── config.py                # Project-wide configuration
├── data/
│   ├── raw/                 # Raw downloaded data + API caches
│   ├── processed/           # Cleaned parquet files
│   └── cache/               # FastF1 cache directory
├── predictions/
│   ├── history/             # JSON prediction records per race
│   └── reports/             # Markdown pre-race prediction reports
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_statistical_analysis.ipynb
│   ├── 03_deep_learning.ipynb
│   └── 04_predictions.ipynb     # Prediction workflow & scorecard
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fastf1_loader.py      # FastF1 telemetry data acquisition
│   │   ├── openf1_loader.py      # OpenF1 API client
│   │   ├── ergast_loader.py      # Jolpica-F1 (Ergast successor) historical data
│   │   └── preprocessing.py      # Data cleaning and feature engineering
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── lap_analysis.py       # Lap time statistical analysis
│   │   ├── telemetry_analysis.py # Telemetry pattern analysis
│   │   └── race_analysis.py      # Race outcome analysis
│   ├── models/
│   │   ├── __init__.py
│   │   ├── datasets.py           # PyTorch datasets for telemetry
│   │   ├── lap_predictor.py      # Lap time prediction model
│   │   ├── driver_classifier.py  # Driving style classification
│   │   └── training.py           # Training loops and utilities
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── track_plots.py        # Circuit visualizations
│   │   ├── telemetry_plots.py    # Telemetry trace plots
│   │   └── dashboard.py          # Summary dashboard generation
│   └── predictions/
│       ├── __init__.py
│       ├── hypotheses.py         # Testable F1 hypotheses
│       ├── registry.py           # Prediction storage and resolution
│       ├── scorecard.py          # Accuracy tracking and calibration
│       ├── backtester.py         # Historical validation engine
│       └── race_predictor.py     # Race weekend prediction orchestrator
├── models/
│   └── checkpoints/              # Saved model weights
└── tests/
    └── __init__.py
```

## Data Sources

### 1. FastF1 (Primary telemetry source)
- **Library**: `fastf1` (pip install)
- **Caching**: Local disk cache at `data/cache/` to avoid re-downloading
- **Coverage**: Seasons 2018–present
- **Key data**: Car telemetry (speed, throttle, brake, DRS, gear, RPM), lap times, weather, tire compounds, GPS car positions
- **Sampling**: High frequency (~300 Hz for some channels)
- **Module**: `src/data/fastf1_loader.py` → `FastF1Loader`

### 2. OpenF1 API (Supplementary real-time/historical data)
- **Base URL**: `https://api.openf1.org/v1/`
- **Auth**: None required for historical data (2023+)
- **Endpoints**: `/car_data`, `/laps`, `/position`, `/pit`, `/weather`, `/race_control`, `/team_radio`
- **Sampling**: ~3.7 Hz telemetry
- **Module**: `src/data/openf1_loader.py` → `OpenF1Client`

### 3. Jolpica-F1 API (Ergast successor for historical results)
- **Base URL**: `https://api.jolpi.ca/ergast/f1/`
- **Coverage**: Historical race results, qualifying, standings, lap times, pit stops back to 1950
- **Format**: JSON, same schema as old Ergast API
- **Module**: `src/data/ergast_loader.py` → `JolpicaClient`

## Data Pipeline

### Preprocessing (`src/data/preprocessing.py`)

1. **Merge** telemetry with lap metadata (compound, lap number, stint, fuel load estimate)
2. **Normalize** telemetry traces to track distance (not time) for cross-driver comparison
3. **Interpolate** missing/corrupt telemetry samples
4. **Engineer features**: sector deltas, braking points, apex speeds, acceleration zones
5. **Create sliding windows** of telemetry for sequence models
6. **Temporal split**: train/val/test by race date (no future data leakage)
7. **Save** processed data as parquet in `data/processed/`

### PyTorch Datasets (`src/models/datasets.py`)

- **TelemetryDataset**: Fixed-length windows of `[Speed, Throttle, Brake, Gear, RPM, DRS]` over track distance, labeled with lap time
- **DriverClassificationDataset**: Same windows, labeled with driver index
- **RaceFeatureDataset**: Aggregated features per driver per race (grid position, qualifying delta, tire strategy, weather)

## Statistical Analysis

### Module: `src/analysis/lap_analysis.py`
1. **Lap time distributions** — KDE, violin, box plots across drivers/compounds. Mann-Whitney U and Kruskal-Wallis tests.
2. **Tire degradation curves** — Polynomial regression of lap time vs. tire life. Compare across compounds and circuits.

### Module: `src/analysis/telemetry_analysis.py`
3. **Telemetry clustering** — Feature extraction + PCA/t-SNE. DTW distance matrices. KMeans and agglomerative clustering of driving styles.

### Module: `src/analysis/race_analysis.py`
4. **Qualifying vs race performance** — Grid-finish Spearman correlation per circuit. Overtaking difficulty index.
5. **Pit strategy analysis** — Strategy effectiveness, undercut/overcut detection.

## Deep Learning Models

### Model 1: Lap Time Predictor (`src/models/lap_predictor.py`)
- **Input**: Telemetry sequence (first N% of lap) → **Output**: Predicted total lap time
- **Architectures**: `LapPredictorCNN` (1D CNN), `LapPredictorLSTM` (BiLSTM), `LapPredictorTransformer` (patch-based Transformer encoder)
- **Training split**: 2018–2023 train, early 2024 val, late 2024 test

### Model 2: Driver Classifier (`src/models/driver_classifier.py`)
- **Input**: Anonymous telemetry trace → **Output**: Driver classification
- **Architecture**: `DriverClassifier` — 1D ResNet with residual blocks
- **Evaluation**: Confusion matrix — which drivers get confused with each other?

### Model 3: Race Outcome Prediction
- **Input**: Pre-race features → **Output**: Finishing position distribution
- **Architecture**: XGBoost baseline, then MLP/attention model
- **Evaluation**: Ranked probability score

## Prediction System (`src/predictions/`)

The core feedback loop of the project. Instead of just analyzing data, the system
generates concrete, testable predictions before each race and scores them after.

### How It Works

1. **Hypotheses** define testable questions with data-driven prediction logic
2. **Backtester** validates hypotheses against historical data (train on 2018-2022, test on 2023-2024)
3. **RacePredictor** generates pre-race predictions for every hypothesis
4. **Registry** stores every prediction with its reasoning and confidence
5. **Scorecard** tracks accuracy over time, shows calibration, identifies which hypotheses work

### Hypotheses (`src/predictions/hypotheses.py`)

| Hypothesis | Question | Key Signals |
|------------|----------|-------------|
| `PoleWinnerHypothesis` | Will the pole sitter win? | Circuit conversion rate, weather risk |
| `RetirementRiskHypothesis` | Which drivers will DNF? | Team reliability, driver history, circuit type |
| `SafetyCarHypothesis` | Will there be a safety car? | Street circuit, weather, historical rate |
| `UndercutHypothesis` | Will undercutting be effective? | Pit loss time, tire degradation rate |
| `WetWeatherUpsetHypothesis` | Will rain cause a major upset? | Rain probability, historical upsets |
| `FirstLapIncidentHypothesis` | Will there be a lap-1 incident? | Circuit T1 risk, qualifying spread |
| `TireStrategyHypothesis` | 1-stop or 2-stop optimal? | Track temp, race length, circuit history |

Each hypothesis:
- Learns from historical data (base rates, circuit-specific patterns)
- Outputs a prediction with confidence score and reasoning
- Can be evaluated against the actual outcome
- Tracks its own accuracy trend over time

### Registry (`src/predictions/registry.py`)

Persistent JSON store. Every prediction gets a unique ID and is saved with:
- The prediction itself (value, confidence, reasoning)
- The hypothesis that generated it
- The actual outcome (filled in post-race)
- Whether it was correct

### Scorecard (`src/predictions/scorecard.py`)

Answers: "Are we getting better?" Provides:
- Accuracy per hypothesis
- Calibration plot (is 70% confidence actually right 70% of the time?)
- Cumulative accuracy trend
- Per-hypothesis rolling accuracy

### Backtester (`src/predictions/backtester.py`)

Validates hypotheses before trusting them. Runs each hypothesis through
historical races and builds a track record. If a hypothesis can't beat
50% on historical data, it needs rework before going live.

### Race Predictor (`src/predictions/race_predictor.py`)

Orchestrates the full workflow:
- `predict_race()` — runs all hypotheses, records predictions, saves markdown report
- `resolve_race()` — scores predictions against outcomes
- `build_pre_race_data()` — assembles qualifying, weather, circuit data
- `build_post_race_outcomes()` — extracts outcomes from results
- `season_report()` — generates season performance summary

### The Feedback Loop

```
Historical Data → Backtest → Identify strong hypotheses
                                    ↓
Upcoming Race → Pre-race data → Predict → Record
                                              ↓
Race Happens → Outcomes → Resolve → Scorecard
                                        ↓
                              Recalibrate / Add new hypotheses
                                        ↓
                              Next Race (repeat)
```

## Training Infrastructure (`src/models/training.py`)

- **Trainer class**: Train/val loop with early stopping, LR scheduling (ReduceLROnPlateau)
- **Gradient clipping**: Max norm 1.0
- **Checkpointing**: Best and final models saved to `models/checkpoints/`
- **Logging**: CSV experiment logs per training run
- **Device**: CPU or CUDA, configurable via `.env`

## Visualization (`src/visualization/`)

- **TrackPlotter**: GPS track maps colored by speed, gear, or any channel
- **TelemetryPlotter**: Multi-channel stacked plots, driver comparisons, speed deltas
- **DashboardGenerator**: Multi-panel session and race summary figures

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Enable FastF1 cache and download 2023-2024 race data
3. Run preprocessing pipeline → parquet files in `data/processed/`
4. Start with `notebooks/01_data_exploration.ipynb` for EDA
5. Run `02_statistical_analysis.ipynb` for tire degradation and clustering
6. Train models in `03_deep_learning.ipynb`
7. **Run `04_predictions.ipynb`** — backtest hypotheses, generate predictions, review scorecard
8. Before each race: run `RacePredictor.predict_race()` with qualifying and weather data
9. After each race: run `RacePredictor.resolve_race()` and check the scorecard

## Notes

- Always use FastF1 caching + local disk cache for API clients
- Telemetry data is large — start with a subset of races
- Some sessions have missing/partial telemetry — error handling is built in
- Temporal train/test split is critical to avoid data leakage
