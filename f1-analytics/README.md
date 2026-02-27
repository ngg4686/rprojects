# F1 Analytics

Hypothesis-driven F1 prediction system. Generates testable pre-race predictions, scores them against reality, and improves over time. Combines statistical analysis and deep learning with a concrete feedback loop — every prediction is recorded, scored, and used to calibrate future predictions.

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and adjust paths if needed (defaults work out of the box).

## Data Sources

| Source | Coverage | Key Data |
|--------|----------|----------|
| **FastF1** | 2018–present | High-frequency telemetry, lap times, weather, tire compounds, GPS positions |
| **OpenF1** | 2023–present | ~3.7 Hz telemetry, pit stops, race control, team radio |
| **Jolpica-F1** | 1950–present | Historical results, standings, qualifying, lap times, pit stops |

## Quick Start

```python
from src.data import JolpicaClient
from src.predictions import RacePredictor

# Load historical data
jolpica = JolpicaClient()
results = jolpica.get_all_results(start_year=2018, end_year=2024)

# Initialize predictor (learns from history)
predictor = RacePredictor(historical_results=results)

# Generate pre-race predictions
quali = jolpica.get_qualifying(2025, 1)
pre_race = predictor.build_pre_race_data(
    qualifying_results=quali,
    circuit_id='bahrain',
    race_laps=57,
    weather_forecast={'rain_probability': 0.05, 'track_temp': 32.0},
)
predictions = predictor.predict_race(2025, 1, 'Bahrain', pre_race)

# After the race — score predictions
race_results = jolpica.get_race_results(2025, 1)
outcomes = predictor.build_post_race_outcomes(race_results)
predictor.resolve_race(2025, 1, outcomes)

# How are we doing?
print(predictor.season_report())
```

## Predictions

The system generates 7 testable predictions before each race:

| Hypothesis | Question |
|------------|----------|
| Pole Winner | Will the pole sitter win? |
| Retirement Risk | Which drivers are most likely to DNF? |
| Safety Car | Will there be a safety car? |
| Undercut | Will undercutting be effective? |
| Wet Weather Upset | Will rain cause a major upset? |
| First Lap Incident | Will there be a lap-1 incident? |
| Tire Strategy | 1-stop or 2-stop optimal? |

Every prediction is recorded with confidence and reasoning, then scored
after the race. The scorecard tracks accuracy by hypothesis, calibration
quality, and trends over time.

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | EDA: data shapes, distributions, sample telemetry, track maps |
| `02_statistical_analysis.ipynb` | Tire degradation, lap time tests, clustering, strategy analysis |
| `03_deep_learning.ipynb` | Driver classification, lap time prediction, training workflows |
| `04_predictions.ipynb` | **Backtest, predict, resolve, and review the scorecard** |

## Project Structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full project architecture, prediction system design, and model descriptions.

## Models

- **Driver Classifier** — 1D ResNet identifying drivers from anonymous telemetry
- **Lap Time Predictor** — CNN/LSTM/Transformer predicting lap time from partial telemetry
- **Race Outcome Predictor** — XGBoost + MLP for finishing position prediction
