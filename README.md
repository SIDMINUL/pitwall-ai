# PitWall AI

**Formula 1 race-strategy analysis and planning dashboard** built with Python and Streamlit.

PitWall AI brings race-session data, pace analysis, weather context, tyre information, candidate strategy generation, and sensitivity analysis into one interactive workspace. It is an actively developed research prototype—not an official F1 product and not a validated race-outcome predictor.

## What it does

- **Race-session exploration:** load supported historical sessions and inspect available lap and race data.
- **Driver and pace analysis:** compare observed lap-time patterns and race-state information when source data supports it.
- **Pre-race planning:** generate candidate pit strategies using configurable race assumptions.
- **Tyre strategy analysis:** estimate compound-specific pace/degradation relationships when tyre compound and tyre-age observations are available.
- **Robustness and sensitivity checks:** compare candidate strategies across specified pit-loss and degradation assumptions.
- **Weather context:** load historical weather context for supported events.
- **Data-quality messaging:** surface missing inputs instead of silently treating unavailable values as known.

Available analyses depend on the selected event, session, provider coverage, and fields returned by each data source.

## Screenshots

### Session overview and historical analysis

The dashboard summarizes the selected session and provides separate views for the race overview, driver and car pace, tyre stints, weather, results and data quality, and strategy simulation.

### Historical strategy analysis

The historical strategy view uses available session tyre observations to estimate compound pace and degradation, compare strategies by pit-stop count, and inspect candidate strategies. These outputs are descriptive estimates; they can be affected by missing tyre data, traffic, fuel load, track evolution, weather, and other race conditions.

### Pre-race strategy planning

When a selected race session is not available yet, the pre-race planner can generate illustrative strategy candidates from configurable race distance, pit-stop loss, stint constraints, and compound assumptions. The assumptions are not official tyre measurements, and estimated times should not be interpreted as verified race forecasts.

### Robustness and sensitivity analysis

The robustness view compares candidate strategies across multiple pit-loss and tyre-degradation scenarios. Scenario ranks and win rates describe only the tested assumptions; they are not real-world probabilities.


## Tech stack

- Python 3.11
- Streamlit
- Pandas and NumPy
- Plotly
- FastF1
- Jolpica API-compatible race results data
- Open-Meteo weather data

## Run locally

### 1. Clone the repository

```powershell
git clone https://github.com/SIDMINUL/pitwall-ai.git
cd pitwall-ai
```

### 2. Create and activate a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate the environment again.

### 3. Install dependencies

```powershell
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 4. Start the dashboard

```powershell
python -m streamlit run app.py
```

Streamlit will print a local URL, normally `http://localhost:8501`.

An internet connection is required for data providers. First loads may take longer while session data is downloaded and cached. The app does not require an OpenAI API key for its current core workflow.

## Run tests and checks

Run the focused automated strategy tests:

```powershell
python -m pytest tests/ -q
python -m compileall -q app.py src
```

The `tests/` directory contains the automated test suite. Data-provider calls and exploratory model investigations are intentionally not run as part of the default test command.

## Repository layout

```text
pitwall-ai/
├── app.py
├── requirements.txt
├── requirements-dev.txt
├── src/
│   └── pitwall/
│       ├── analytics/    # pace, race state, planning and strategy models
│       └── data/         # session, results, circuit, tyre and weather loaders
├── tests/                # automated strategy tests
├── docs/                 # development notes
├── research/             # exploratory model and data validation utilities
│   ├── README.md
│   └── archive/           # older single-race experiments
├── README.md
├── CONTRIBUTING.md
└── LICENSE
```

## Model limitations

- Strategy rankings are estimates from simplified assumptions and available observations; they are not guaranteed race outcomes.
- Linear tyre-age fits are descriptive. Lap times are affected by driver pace, fuel load, traffic, track evolution, weather, setup, and session conditions.
- Fit statistics such as R² do not establish causality or prove out-of-sample accuracy.
- Sensitivity results reflect the scenarios explicitly tested; they are not real-world probability estimates.
- Pit-stop rules, tyre allocation requirements, safety cars, virtual safety cars, traffic, and strategic interactions may not be fully represented.
- Some race-result sources do not provide tyre compound or tyre-age data. Tyre-model analysis should report unavailable when required inputs are missing.

Use this project for learning and exploratory analysis, not operational race decisions.

## Data sources and attribution

The app may retrieve public motorsport data from [FastF1](https://docs.fastf1.dev/), [Jolpica](https://github.com/jolpica/jolpica-f1), and weather information from [Open-Meteo](https://open-meteo.com/). Refer to each provider's documentation and terms for usage and attribution requirements. PitWall AI is an independent project and is not affiliated with Formula 1, the FIA, or any team.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## License

Distributed under the MIT License; see [LICENSE](LICENSE).
