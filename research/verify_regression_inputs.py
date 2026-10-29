
import numpy as np
import pandas as pd

LAPS_PATH = "data/openf1/multirace_tyre_laps.csv"
SLOPES_PATH = "data/openf1/multirace_stint_slopes.csv"

df = pd.read_csv(LAPS_PATH)
slopes = pd.read_csv(SLOPES_PATH)

for col in [
    "LapNumber", "LapTimeSeconds", "TyreLife",
    "StintNumber", "Sector1Seconds",
    "Sector2Seconds", "Sector3Seconds",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Reproduce the filters in analyze_multirace_degradation.py.
if "IsPitOutLap" in df.columns:
    pit_out = (
        df["IsPitOutLap"].astype(str).str.lower()
        .isin(["true", "1"])
    )
    df = df[~pit_out].copy()

df = df.dropna(subset=[
    "Country", "SessionKey", "DriverNumber", "StintNumber",
    "LapNumber", "Compound", "TyreLife", "LapTimeSeconds",
])

df = df[
    df["TyreLife"].between(1, 80)
    & df["LapTimeSeconds"].between(60, 115)
].copy()

sector_cols = [
    "Sector1Seconds", "Sector2Seconds", "Sector3Seconds"
]

if all(c in df.columns for c in sector_cols):
    complete = df[sector_cols].notna().all(axis=1)
    sector_sum = df[sector_cols].sum(axis=1, min_count=3)
    discrepancy = (sector_sum - df["LapTimeSeconds"]).abs()
    df = df[~complete | (discrepancy <= 0.5)].copy()

print("Filtered laps:", len(df))

worst = slopes.nsmallest(5, "SlopeSecondsPerLap")

for row in worst.itertuples(index=False):
    group = df[
        (df["Country"] == row.Country)
        & (df["SessionKey"] == row.SessionKey)
        & (df["DriverNumber"] == row.DriverNumber)
        & (df["StintNumber"] == row.StintNumber)
        & (df["Compound"] == row.Compound)
    ].sort_values("TyreLife").copy()

    print("\n" + "=" * 80)
    print(
        f"{row.Country} | Driver {row.DriverNumber} | "
        f"Stint {row.StintNumber} | {row.Compound}"
    )
    print("Stored slope:", round(row.SlopeSecondsPerLap, 4))
    print("Filtered rows:", len(group))

    if len(group) < 5 or group["TyreLife"].nunique() < 4:
        print("Insufficient filtered data to refit.")
        continue

    # Repeat the same within-stint 10th/90th percentile trim.
    low = group["LapTimeSeconds"].quantile(0.10)
    high = group["LapTimeSeconds"].quantile(0.90)
    group = group[
        group["LapTimeSeconds"].between(low, high)
    ].copy()

    if len(group) < 5 or group["TyreLife"].nunique() < 4:
        print("Insufficient data after trimming.")
        continue

    slope, intercept = np.polyfit(
        group["TyreLife"].to_numpy(float),
        group["LapTimeSeconds"].to_numpy(float),
        1,
    )

    print("Rows actually used:", len(group))
    print("Recomputed slope:", round(float(slope), 4))
    print(
        group[
            ["LapNumber", "LapTimeSeconds", "TyreLife"]
        ].to_string(index=False)
    )
