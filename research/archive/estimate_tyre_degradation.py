
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("data/openf1/bahrain_2024_tyre_laps.csv")
OUTPUT = Path("data/openf1/stint_degradation_estimates.csv")

df = pd.read_csv(INPUT)

# Keep plausible race lap times and positive tyre ages.
df["LapTimeSeconds"] = pd.to_numeric(
    df["LapTimeSeconds"], errors="coerce"
)
df["TyreLife"] = pd.to_numeric(df["TyreLife"], errors="coerce")

df = df.dropna(
    subset=[
        "DriverNumber",
        "LapNumber",
        "LapTimeSeconds",
        "Compound",
        "TyreLife",
        "stint_number",
    ]
)

df = df[
    df["TyreLife"].between(1, 80)
    & df["LapTimeSeconds"].between(60, 115)
].copy()

# Exclude pit-out laps when the source flag is available.
if "is_pit_out_lap" in df.columns:
    pit_out = df["is_pit_out_lap"].astype(str).str.lower().isin(
        ["true", "1"]
    )
    df = df[~pit_out].copy()

# Estimate one slope per driver and stint.
# Require at least 5 observations and 4 distinct tyre ages.
estimates = []

for (driver, stint, compound), group in df.groupby(
    ["DriverNumber", "stint_number", "Compound"]
):
    group = group.sort_values("TyreLife").copy()

    if len(group) < 5 or group["TyreLife"].nunique() < 4:
        continue

    # Trim extreme lap times within this stint.
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
        "DegradationSecondsPerLap": slope,
        "Intercept": intercept,
    })

stints = pd.DataFrame(estimates)

if stints.empty:
    print("No stints met the minimum data requirements.")
    raise SystemExit(1)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
stints.to_csv(OUTPUT, index=False)

print("\nSTINTS ANALYZED")
print(len(stints))

print("\nESTIMATES BY COMPOUND")
summary = stints.groupby("Compound").agg(
    Stints=("DegradationSecondsPerLap", "size"),
    MedianSlope=("DegradationSecondsPerLap", "median"),
    MeanSlope=("DegradationSecondsPerLap", "mean"),
    SlopeStdDev=("DegradationSecondsPerLap", "std"),
    MedianSamples=("Samples", "median"),
)

print(summary.round(4).to_string())

print("\nNEGATIVE SLOPES BY COMPOUND")
print(
    stints.assign(
        Negative=stints["DegradationSecondsPerLap"] < 0
    )
    .groupby("Compound")["Negative"]
    .mean()
    .mul(100)
    .round(1)
    .astype(str)
    .add("%")
    .to_string()
)

print("\nSaved:", OUTPUT.resolve())
print(
    "\nCaution: these are exploratory within-stint trends, "
    "not causal tyre degradation estimates."
)
