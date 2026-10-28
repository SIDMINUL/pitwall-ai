
from pathlib import Path

import pandas as pd

output_dir = Path.cwd()

pace_path = output_dir / "recent_driver_pace.csv"
strategy_path = output_dir / "race_strategy_candidates.csv"

if not pace_path.exists() or not strategy_path.exists():
    raise FileNotFoundError(
        "CSV files not found. Run test_race_planner.py first."
    )

pace = pd.read_csv(pace_path)
strategies = pd.read_csv(strategy_path)

print("=== FILE SHAPES ===")
print("Driver pace:", pace.shape)
print("Strategy candidates:", strategies.shape)

print("\n=== PACE DATA QUALITY ===")
print("Missing values:")
print(pace.isna().sum().to_string())

if "ValidLaps" in pace.columns:
    print("\nValid laps per driver:")
    print(
        pace[["DriverId", "ValidLaps"]]
        .sort_values("ValidLaps")
        .to_string(index=False)
    )

print("\n=== STRATEGY DATA QUALITY ===")
print("Missing values:")
print(strategies.isna().sum().to_string())

print("\nUnique strategy counts:")
for column in [
    "DriverId",
    "Stops",
    "CompoundSequence",
    "StintLengths",
    "ModelStatus",
]:
    if column in strategies.columns:
        print(f"{column}: {strategies[column].nunique()}")

print("\n=== BEST STRATEGY PER DRIVER AND STOP COUNT ===")
if "EstimatedRaceTimeSeconds" in strategies.columns:
    best = (
        strategies.sort_values("EstimatedRaceTimeSeconds")
        .groupby(["DriverId", "Stops"], as_index=False)
        .first()
    )

    columns = [
        "DriverId",
        "Stops",
        "CompoundSequence",
        "StintLengths",
        "EstimatedRaceTimeSeconds",
        "ModelStatus",
    ]
    columns = [c for c in columns if c in best.columns]
    print(best[columns].to_string(index=False))

print("\n=== STRATEGY TIME RANGE ===")
if "EstimatedRaceTimeSeconds" in strategies.columns:
    print(
        strategies.groupby("Stops")["EstimatedRaceTimeSeconds"]
        .agg(["min", "max", "count"])
        .round(2)
        .to_string()
    )

print("\nInspection complete.")
