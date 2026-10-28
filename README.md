# PitWall AI

An evidence-grounded Formula 1 race intelligence assistant combining historical race data, deterministic analytics, simulation, and GenAI.

## Initial scope

The first milestone is deliberately small: load a completed F1 session, inspect available data, calculate basic driver/pace metrics, and display them in a Streamlit dashboard. We will add weather-aware scenarios, tyre strategy simulation, RAG over regulations, and tool-using agents in later milestones.

## Requirements

- Python 3.11 recommended
- Internet connection for downloading FastF1 session data
- Ollama is optional for the first milestone

## Windows setup (PowerShell)

From `C:\my studies\projects`:

```powershell
mkdir pitwall-ai
cd pitwall-ai
# Extract the provided starter files into this folder if using the ZIP.
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Data/cache

FastF1 caches downloaded session data under `data/cache/`. Do not commit cache files or downloaded data.

## Project structure

```text
pitwall-ai/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── src/
│   └── pitwall/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── fastf1_loader.py
│       └── analytics/
│           ├── __init__.py
│           └── pace.py
└── data/
    └── cache/
```

## Initial data-quality rules

- Never silently treat missing telemetry as zero.
- Label derived values as estimates, not ground truth.
- Distinguish observed data, calculated metrics, and LLM-generated explanations.
- A strategy recommendation must include assumptions and uncertainty.
