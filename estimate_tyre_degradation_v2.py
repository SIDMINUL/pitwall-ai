
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("data/openf1/bahrain_2024_tyre_laps.csv")
OUTPUT = Path("data/openf1/stint_degradation_v2.csv")

df = pd.read_csv(INPUT)

numeric_columns = [
    "LapNumber",
    "LapTimeSeconds",
    "TyreLife",
    "stint_number",
    "duration_sector_1",
    "duration_sector_2",
    "duration_sector_3",
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

required = [
    "DriverNumber",
    "LapNumber",
    "LapTimeSeconds",
    "Compound",
    "TyreLife",
    "stint_number",
]

missing = set(required) - set(df.columns)
if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}. "
        "Rebuild the CSV from the OpenF1 laps and stints endpoints, "
        "including the sector durations and pit-out flag."
    )

# Exclude pit-out laps when the flag is present.
if "is_pit_out_lap" in df.columns:
    pit_out = (
        df["is_pit_out_lap"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )
    df = df[~pit_out].copy()

df = df.dropna(subset=required)
df = df[
    df["TyreLife"].between(1, 80)
    & df["LapTimeSeconds"].between(60, 115)
].copy()

# Where all three sectors are available, exclude rows whose sector
# total differs substantially from the reported lap duration.
sector_cols = [
    "duration_sector_1",
    "duration_sector_2",
    "duration_sector_3",
]

if all(col in df.columns for col in sector_cols):
    complete = df[sector_cols].notna().all(axis=1)
    sector_sum = df[sector_cols].sum(axis=1, min_count=3)
    discrepancy = (sector_sum - df["LapTimeSeconds"]).abs()

    # A tolerance is necessary because of timing precision.
    df = df[~complete | (discrepancy <= 0.5)].copy()

estimates = []

for (driver, stint, compound), group in df.groupby(
    ["DriverNumber", "stint_number", "Compound"]
):
    group = group.sort_values("TyreLife")

    if len(group) < 5 or group["TyreLife"].nunique() < 4:
        continue

    # Trim extreme times within each stint.
    low = group["LapTimeSeconds"].quantile(0.10)
    high = group["LapTimeSeconds"].quantile(0.90)
    group = group[
        group["LapTimeSeconds"].between(low, high)
    ]

    if len(group) < 5 or group["TyreLife"].nunique() < 4:
        continue

    x = group["TyreLife"].to_numpy(dtype=float)
    y = group["LapTimeSeconds"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)

    estimates.append({
        "DriverNumber": driver,
        "StintNumber": stint,
        "Compound": compound,
        "Samples": len(group),
        "UniqueTyreAges": group["TyreLife"].nunique(),
        "SlopeSecondsPerLap": slope,
        "Intercept": intercept,
    })

stints = pd.DataFrame(estimates)

if stints.empty:
    raise RuntimeError("No stints met the data requirements.")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
stints.to_csv(OUTPUT, index=False)

print("Usable laps:", len(df))
print("Stints analyzed:", len(stints))
print("\nCompound summary:")

summary = stints.groupby("Compound").agg(
    Stints=("SlopeSecondsPerLap", "size"),
    MedianSlope=("SlopeSecondsPerLap", "median"),
    MeanSlope=("SlopeSecondsPerLap", "mean"),
    SlopeStdDev=("SlopeSecondsPerLap", "std"),
    NegativeSlopeShare=(
        "SlopeSecondsPerLap",
        lambda s: (s < 0).mean() * 100,
    ),
)

print(summary.round(4).to_string())
print("\nSaved:", OUTPUT.resolve())
print(
    "\nInterpretation: exploratory lap-time trends, not causal "
    "tyre degradation estimates. Fuel, traffic, and track evolution "
    "remain potential confounders."
)
