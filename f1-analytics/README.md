# F1 Analytics

Formula 1 data analysis project combining statistical analysis and deep learning. Pulls from multiple F1 data sources (FastF1, OpenF1, Jolpica-F1), builds a clean data pipeline, and provides tools for exploratory analysis and model training.

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
from src.data import FastF1Loader, JolpicaClient

# Load a race session
loader = FastF1Loader()
session = loader.get_session(2024, 'Monza', 'R')
laps = session.laps

# Get historical results
jolpica = JolpicaClient()
results = jolpica.get_race_results(2024)
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | EDA: data shapes, distributions, sample telemetry, track maps |
| `02_statistical_analysis.ipynb` | Tire degradation, lap time tests, clustering, strategy analysis |
| `03_deep_learning.ipynb` | Driver classification, lap time prediction, training workflows |

## Project Structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full project architecture, data pipeline details, and model descriptions.

## Models

- **Driver Classifier** — 1D ResNet identifying drivers from anonymous telemetry
- **Lap Time Predictor** — CNN/LSTM/Transformer predicting lap time from partial telemetry
- **Race Outcome Predictor** — XGBoost + MLP for finishing position prediction
