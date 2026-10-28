# PitWall AI

PitWall AI is a Formula 1 race-strategy analysis application built with Python and Streamlit. It brings race-session data, strategy exploration, and scenario-based comparison into one interface.

> **Project status:** Active development / research prototype. Strategy outputs are simplified estimates for exploration, not validated race predictions or professional motorsport advice.

## Features

- **Race-session exploration:** Load and inspect available session and lap data.
- **Pre-race strategy planner:** Explore candidate tyre sequences and stint lengths using configurable assumptions.
- **Strategy simulation:** Compare candidate pit-stop plans and estimated running times.
- **Tyre-model diagnostics:** Where compatible compound and tyre-age data is available, fit simple compound-specific linear models and expose sample and fit diagnostics.
- **Robustness analysis:** Compare strategy rankings across selected pit-stop-loss and degradation assumptions.
- **Data-quality messaging:** Distinguish data available from a source from fields that are not present.

Availability depends on the selected session and data source. For example, a lap-time dataset without tyre compound and tyre-age columns cannot support the historical tyre model.

## Screenshots

Add current screenshots to `docs/images/` and embed them here when ready. Recommended captures:
- Main dashboard / session selection
- Pre-race strategy planner
- Historical strategy candidates
- Strategy robustness analysis

## Architecture

The application is organized around a Streamlit UI and Python analytics modules:

```text
pitwall-ai/
├── app.py
├── src/
│   └── pitwall/
│       ├── analytics/       # Strategy modelling and simulation
│       └── ...              # Data loaders and supporting modules
├── tests/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

The exact module layout may evolve as the project develops.

## Getting started

### Requirements

- Python 3.11 recommended (the current development environment uses Python 3.11)
- Git
- Internet access for data sources that fetch remote session/weather information

### 1. Clone the repository

```bash
git clone https://github.com/SIDMINUL/pitwall-ai.git
cd pitwall-ai
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

From the repository root, on Windows PowerShell:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m streamlit run app.py
```

Then open the local URL printed by Streamlit, normally `http://localhost:8501`.

If the source tree uses additional environment variables, copy `.env.example` to a local `.env` and fill in only the values you need. Never commit secrets.

## Tests and quality checks

Run the available tests:

```bash
python -m pytest -q
```

Compile-check the main application and strategy module:

```bash
python -m compileall -q app.py src
```

The current development test suite has passed four tests, but pytest collection has exhibited unusually long runtimes in the development environment. That performance issue remains under investigation; passing tests do not establish the predictive accuracy of the strategy model.

## Strategy model: scope and limitations

The current tyre model fits a simple linear relationship between lap time and tyre age for each compound. Strategy estimates can be affected by confounding factors including driver pace, fuel load, traffic, track evolution, weather, safety cars, setup, and race circumstances.

The robustness analysis varies selected assumptions. It is a **sensitivity analysis**, not a probability model: a scenario win rate is the fraction of tested scenarios in which a candidate ranks first, not the probability of winning a real race.

The optimizer should not be described as a validated race-time predictor until it has been tested against appropriate historical data, compared with baselines, and evaluated with reproducible metrics.

## Data sources and attribution

The project may use public Formula 1 timing/session data and external weather or schedule services depending on the selected loader. Review the terms and attribution requirements of each source before redistributing data or deploying the application. This repository does not claim ownership of third-party data, trademarks, team names, or Formula 1 branding.

## Roadmap

- [ ] Reproducible dependency lock and clean-environment setup test
- [ ] Add synthetic unit tests for optimizer feasibility and scenario ranking
- [ ] Investigate slow pytest collection
- [ ] Improve pace/degradation separation and document uncertainty
- [ ] Validate strategy rankings against historical race outcomes
- [ ] Add screenshots and a short demo
- [ ] Add continuous integration for linting, syntax checks, and tests

## Contributing

Bug reports and focused pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the MIT License; see [LICENSE](LICENSE).

---

Built as an independent Formula 1 strategy-analysis and engineering project.
