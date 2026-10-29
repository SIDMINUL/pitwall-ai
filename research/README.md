# Research and experimental scripts

These scripts support the tyre-degradation and pace-loss analysis behind PitWall AI. They are research utilities, not part of the Streamlit app's runtime.

## Multi-race workflow

Run commands from the repository root after installing the project requirements and preparing the local OpenF1 CSV inputs.

1. `build_multirace_dataset.py` — downloads/builds a multi-race tyre-lap dataset.
2. `analyze_multirace_degradation.py` — estimates within-stint lap-time slopes across races.
3. `compare_degradation_models.py` — compares basic tyre-age slopes with a model that also accounts for race lap progression.
4. `estimate_within_stint_pace_loss.py` — derives pace loss relative to early-stint laps.
5. `validate_pace_loss_by_race.py` — evaluates pace-loss estimates using race-level holdouts.
6. `verify_regression_inputs.py` — diagnostic tool for inspecting regression inputs and unusual slopes.

The scripts expect generated CSV files under `data/openf1/`. These files are local data artifacts and are not included in the repository. Check each script's input paths and options before running; the workflow has not been verified as a single end-to-end pipeline on a clean machine.

## Archived experiments

`archive/estimate_tyre_degradation.py` and `archive/estimate_tyre_degradation_v2.py` are older Bahrain 2024 single-race estimators retained for reproducibility and comparison. Prefer the multi-race workflow for ongoing research.

## Interpretation

These are exploratory analyses of historical timing data. Tyre-age slopes and derived pace-loss curves can be affected by traffic, fuel load, track evolution, weather, safety cars, driver behavior, and data quality. They should not be presented as validated race predictions without further testing.
