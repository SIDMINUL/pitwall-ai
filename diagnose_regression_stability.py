
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/openf1/multirace_tyre_laps.csv")

df = pd.read_csv(DATA_PATH)

numeric_columns = [
    "SessionKey",
    "DriverNumber",
    "StintNumber",
    "LapNumber",
    "TyreLife",
    "LapTimeSeconds",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=[
        "Country", "SessionKey", "DriverNumber",
        "StintNumber", "Compound", "LapNumber",
        "TyreLife", "LapTimeSeconds",
    ]
)

df = df[
    df["TyreLife"].between(1, 80)
    & df["LapTimeSeconds"].between(60, 115)
].copy()

if "IsPitOutLap" in df.columns:
    pit_out = df["IsPitOutLap"].astype(str).str.lower().isin(
        ["true", "1"]
    )
    df = df.loc[~pit_out].copy()

keys = [
    "Country", "SessionKey", "DriverNumber",
    "StintNumber", "Compound",
]

results = []

for group_key, group in df.groupby(keys):
    group = group.sort_values("LapNumber")

    if len(group) < 5 or group["TyreLife"].nunique() < 4:
        continue

    low = group["LapTimeSeconds"].quantile(0.10)
    high = group["LapTimeSeconds"].quantile(0.90)
    group = group[
        group["LapTimeSeconds"].between(low, high)
    ]

    if len(group) < 5:
        continue

    age = group["TyreLife"].to_numpy(dtype=float)
    lap = group["LapNumber"].to_numpy(dtype=float)
    time = group["LapTimeSeconds"].to_numpy(dtype=float)

    age_centered = age - age.mean()
    lap_centered = lap - lap.mean()

    correlation = np.corrcoef(age, lap)[0, 1]

    # VIF for tyre age in a two-predictor linear model.
    # With only these two predictors, VIF = 1 / (1 - r^2).
    vif = (
        1 / (1 - correlation**2)
        if abs(correlation) < 1
        else np.inf
    )

    X = np.column_stack([
        np.ones(len(group)),
        age_centered,
        lap_centered,
    ])

    coefficients, *_ = np.linalg.lstsq(X, time, rcond=None)

    results.append({
        "Country": group_key[0],
        "Compound": group_key[4],
        "Samples": len(group),
        "AgeLapCorrelation": correlation,
        "TyreAgeVIF": vif,
        "AdjustedAgeSlope": coefficients[1],
    })

out = pd.DataFrame(results)

print("Analyzed stints:", len(out))
print("\nCorrelation and VIF summary:")
print(
    out[["AgeLapCorrelation", "TyreAgeVIF"]]
    .describe()
    .round(3)
    .to_string()
)

print("\nHighest-VIF stints:")
print(
    out.sort_values("TyreAgeVIF", ascending=False)
    .head(15)
    .round(4)
    .to_string(index=False)
)

print("\nMedian VIF by compound:")
print(
    out.groupby("Compound")["TyreAgeVIF"]
    .median()
    .round(3)
    .to_string()
)

output_path = Path("data/openf1/regression_stability.csv")
out.to_csv(output_path, index=False)
print(f"\nSaved: {output_path}")
