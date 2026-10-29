
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/openf1/multirace_tyre_laps.csv")
OUTPUT_PATH = Path("data/openf1/degradation_model_comparison.csv")

df = pd.read_csv(DATA_PATH)

numeric_columns = [
    "SessionKey",
    "DriverNumber",
    "StintNumber",
    "LapNumber",
    "TyreLife",
    "LapTimeSeconds",
]

for column in numeric_columns:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

required = [
    "Country",
    "SessionKey",
    "DriverNumber",
    "StintNumber",
    "Compound",
    "LapNumber",
    "TyreLife",
    "LapTimeSeconds",
]

df = df.dropna(subset=required).copy()
df = df[
    df["TyreLife"].between(1, 80)
    & df["LapTimeSeconds"].between(60, 115)
].copy()

if "IsPitOutLap" in df.columns:
    pit_out = (
        df["IsPitOutLap"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )
    df = df.loc[~pit_out].copy()

results = []

group_columns = [
    "Country",
    "SessionKey",
    "DriverNumber",
    "StintNumber",
    "Compound",
]

for keys, stint in df.groupby(group_columns):
    stint = stint.sort_values("LapNumber").copy()

    if len(stint) < 5 or stint["TyreLife"].nunique() < 4:
        continue

    # Apply the same within-stint 10th/90th percentile trim
    low = stint["LapTimeSeconds"].quantile(0.10)
    high = stint["LapTimeSeconds"].quantile(0.90)
    trimmed = stint[
        stint["LapTimeSeconds"].between(low, high)
    ].copy()

    if len(trimmed) < 5:
        continue

    tyre_age = trimmed["TyreLife"].to_numpy(dtype=float)
    lap_time = trimmed["LapTimeSeconds"].to_numpy(dtype=float)
    lap_number = trimmed["LapNumber"].to_numpy(dtype=float)

    # Model A: lap time explained by tyre age alone
    slope_age = np.polyfit(tyre_age, lap_time, 1)[0]

    # Model B: control for race lap progress.
    # This is a diagnostic, not a complete fuel-load correction.
    X = np.column_stack([
        np.ones(len(trimmed)),
        tyre_age,
        lap_number,
    ])

    coefficients, *_ = np.linalg.lstsq(X, lap_time, rcond=None)
    adjusted_age_slope = coefficients[1]

    results.append({
        "Country": keys[0],
        "SessionKey": keys[1],
        "DriverNumber": keys[2],
        "StintNumber": keys[3],
        "Compound": keys[4],
        "Samples": len(trimmed),
        "TyreAgeSlope": slope_age,
        "RaceProgressAdjustedSlope": adjusted_age_slope,
    })

result = pd.DataFrame(results)

if result.empty:
    raise RuntimeError("No stints passed the filtering requirements.")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_PATH, index=False)

print(f"Stints analyzed: {len(result)}")
print(f"Saved: {OUTPUT_PATH}")

print("\nMedian slope by compound:")
print(
    result.groupby("Compound")[
        ["TyreAgeSlope", "RaceProgressAdjustedSlope"]
    ].median().round(4).to_string()
)

print("\nMedian slope by race and compound:")
print(
    result.groupby(["Country", "Compound"])[
        ["TyreAgeSlope", "RaceProgressAdjustedSlope"]
    ].median().round(4).to_string()
)

print("\nNegative slope proportions:")
print(
    result.assign(
        OriginalNegative=result["TyreAgeSlope"] < 0,
        AdjustedNegative=result["RaceProgressAdjustedSlope"] < 0,
    )
    .groupby(["Country", "Compound"])[
        ["OriginalNegative", "AdjustedNegative"]
    ]
    .mean()
    .round(3)
    .to_string()
)
