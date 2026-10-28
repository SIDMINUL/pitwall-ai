
from pathlib import Path

import numpy as np
import pandas as pd

path = Path("data/openf1/multirace_tyre_laps.csv")
df = pd.read_csv(path)

for column in ["LapNumber", "TyreLife", "LapTimeSeconds"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(
    subset=[
        "Country",
        "SessionKey",
        "DriverNumber",
        "StintNumber",
        "LapNumber",
        "TyreLife",
        "LapTimeSeconds",
    ]
)

if "IsPitOutLap" in df.columns:
    pit_out = (
        df["IsPitOutLap"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )
    df = df.loc[~pit_out].copy()

df = df[
    df["TyreLife"].between(1, 80)
    & df["LapTimeSeconds"].between(60, 115)
].copy()

keys = [
    "Country",
    "SessionKey",
    "DriverNumber",
    "StintNumber",
]

df["AgeMinusLap"] = df["TyreLife"] - df["LapNumber"]

print("Rows:", len(df))
print("\nAgeMinusLap distribution:")
print(df["AgeMinusLap"].describe().round(3).to_string())

print(
    "\nMost common AgeMinusLap values:"
)
print(df["AgeMinusLap"].value_counts().head(10).to_string())

stints = []

for key, group in df.groupby(keys):
    if len(group) < 2:
        continue

    differences = group["AgeMinusLap"].dropna()

    stints.append({
        "Country": key[0],
        "Samples": len(group),
        "UniqueAgeMinusLap": differences.nunique(),
        "AgeLapCorrelation": group["TyreLife"].corr(
            group["LapNumber"]
        ),
        "AgeLapDifference": differences.median(),
    })

summary = pd.DataFrame(stints)

print("\nStint relationship summary:")
print(
    summary[
        ["AgeLapCorrelation", "UniqueAgeMinusLap"]
    ].describe().round(4).to_string()
)

print("\nStints with non-constant age-minus-lap:")
print(
    summary[summary["UniqueAgeMinusLap"] > 1]
    .head(20)
    .to_string(index=False)
)

out = Path("data/openf1/age_lap_relationship.csv")
summary.to_csv(out, index=False)
print(f"\nSaved: {out}")
