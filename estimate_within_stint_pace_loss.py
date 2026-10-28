
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("data/openf1/multirace_tyre_laps.csv")
OUTPUT = Path("data/openf1/within_stint_pace_loss.csv")
SUMMARY = Path("data/openf1/within_stint_pace_loss_summary.csv")

df = pd.read_csv(INPUT)

for col in ["SessionKey", "DriverNumber", "StintNumber",
            "LapNumber", "TyreLife", "LapTimeSeconds"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

required = [
    "Country", "SessionKey", "DriverNumber", "StintNumber",
    "Compound", "LapNumber", "TyreLife", "LapTimeSeconds",
]
df = df.dropna(subset=required).copy()

df = df[
    df["TyreLife"].between(1, 80)
    & df["LapTimeSeconds"].between(60, 115)
].copy()

if "IsPitOutLap" in df.columns:
    pit_out = (
        df["IsPitOutLap"].astype(str).str.lower().isin(["true", "1"])
    )
    df = df.loc[~pit_out].copy()

group_cols = [
    "Country", "SessionKey", "DriverNumber", "StintNumber", "Compound"
]

results = []

for keys, group in df.groupby(group_cols):
    group = group.sort_values("TyreLife").copy()

    # Need at least 6 clean laps and several tyre ages.
    if len(group) < 6 or group["TyreLife"].nunique() < 5:
        continue

    # Use the first 3 available tyre-age observations as the baseline.
    baseline_rows = group.nsmallest(3, "TyreLife")
    baseline = baseline_rows["LapTimeSeconds"].median()

    group["PaceLossSeconds"] = group["LapTimeSeconds"] - baseline

    # Summarize pace loss by tyre age; this is descriptive, not causal.
    for age, age_group in group.groupby("TyreLife"):
        results.append({
            "Country": keys[0],
            "SessionKey": keys[1],
            "DriverNumber": keys[2],
            "StintNumber": keys[3],
            "Compound": keys[4],
            "TyreLife": float(age),
            "LapsAtAge": len(age_group),
            "BaselineSeconds": float(baseline),
            "MedianPaceLossSeconds": float(
                age_group["PaceLossSeconds"].median()
            ),
        })

out = pd.DataFrame(results)
if out.empty:
    raise RuntimeError("No stints met the sample requirements.")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT, index=False)

summary = (
    out.groupby(["Country", "Compound", "TyreLife"])
    .agg(
        Stints=("StintNumber", "nunique"),
        MedianPaceLossSeconds=("MedianPaceLossSeconds", "median"),
        MeanPaceLossSeconds=("MedianPaceLossSeconds", "mean"),
    )
    .reset_index()
)

summary.to_csv(SUMMARY, index=False)

print(f"Stint-age observations: {len(out)}")
print(f"Saved: {OUTPUT}")
print(f"Saved: {SUMMARY}")

print("\nMedian observed pace loss by race and compound:")
print(
    out.groupby(["Country", "Compound"])["MedianPaceLossSeconds"]
    .median()
    .round(3)
    .to_string()
)

print("\nPooled observed pace loss by compound and tyre age:")
print(
    summary.groupby(["Compound", "TyreLife"])
    ["MedianPaceLossSeconds"]
    .median()
    .unstack()
    .round(2)
    .to_string()
)
