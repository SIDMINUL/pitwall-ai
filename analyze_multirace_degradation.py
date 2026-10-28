
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("data/openf1/multirace_tyre_laps.csv")
OUTPUT_STINTS = Path("data/openf1/multirace_stint_slopes.csv")
OUTPUT_SUMMARY = Path("data/openf1/multirace_degradation_summary.csv")

df = pd.read_csv(INPUT)

# Convert numeric fields safely.
for col in ["LapNumber", "LapTimeSeconds", "TyreLife", "StintNumber"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Exclude pit-out laps and invalid tyre ages.
if "IsPitOutLap" in df.columns:
    pit_out = (
        df["IsPitOutLap"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )
    df = df[~pit_out].copy()

df = df.dropna(
    subset=[
        "Country", "SessionKey", "DriverNumber", "StintNumber",
        "LapNumber", "Compound", "TyreLife", "LapTimeSeconds",
    ]
)

df = df[
    df["TyreLife"].between(1, 80)
    & df["LapTimeSeconds"].between(60, 115)
].copy()

# Exclude inconsistent sector totals when all three sectors exist.
sector_cols = [
    "Sector1Seconds", "Sector2Seconds", "Sector3Seconds"
]

if all(col in df.columns for col in sector_cols):
    complete = df[sector_cols].notna().all(axis=1)
    sector_sum = df[sector_cols].sum(axis=1, min_count=3)
    discrepancy = (sector_sum - df["LapTimeSeconds"]).abs()
    df = df[~complete | (discrepancy <= 0.5)].copy()

group_cols = [
    "Country", "SessionKey", "DriverNumber",
    "StintNumber", "Compound",
]

records = []

for keys, group in df.groupby(group_cols):
    group = group.sort_values("TyreLife")

    if len(group) < 5 or group["TyreLife"].nunique() < 4:
        continue

    # Robustly trim extreme times within this stint.
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

    records.append({
        "Country": keys[0],
        "SessionKey": keys[1],
        "DriverNumber": keys[2],
        "StintNumber": keys[3],
        "Compound": keys[4],
        "Samples": len(group),
        "UniqueTyreAges": group["TyreLife"].nunique(),
        "SlopeSecondsPerLap": slope,
        "Intercept": intercept,
    })

stints = pd.DataFrame(records)

if stints.empty:
    raise RuntimeError("No stints passed the minimum data requirements.")

OUTPUT_STINTS.parent.mkdir(parents=True, exist_ok=True)
stints.to_csv(OUTPUT_STINTS, index=False)

summary = (
    stints.groupby(["Country", "Compound"])
    .agg(
        Stints=("SlopeSecondsPerLap", "size"),
        MedianSlope=("SlopeSecondsPerLap", "median"),
        MeanSlope=("SlopeSecondsPerLap", "mean"),
        SlopeStdDev=("SlopeSecondsPerLap", "std"),
        NegativeSlopePct=(
            "SlopeSecondsPerLap",
            lambda x: 100 * (x < 0).mean(),
        ),
        MedianSamples=("Samples", "median"),
    )
    .reset_index()
)

summary.to_csv(OUTPUT_SUMMARY, index=False)

print("USABLE LAP RECORDS:", len(df))
print("STINTS ANALYZED:", len(stints))

print("\nSTINT COUNTS BY RACE AND COMPOUND")
print(
    stints.groupby(["Country", "Compound"])
    .size()
    .unstack(fill_value=0)
    .to_string()
)

print("\nDEGRADATION SUMMARY")
print(summary.round(4).to_string(index=False))

print("\nSaved stint estimates:", OUTPUT_STINTS.resolve())
print("Saved summary:", OUTPUT_SUMMARY.resolve())

print(
    "\nCaution: these are observational lap-time trends. "
    "Fuel, traffic, track evolution, and race conditions are not "
    "fully controlled, so slopes are not causal tyre-wear estimates."
)
