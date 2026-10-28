
import pandas as pd

laps_path = "data/openf1/multirace_tyre_laps.csv"
slopes_path = "data/openf1/multirace_stint_slopes.csv"

laps = pd.read_csv(laps_path)
slopes = pd.read_csv(slopes_path)

print("\n1. NEGATIVE SLOPES BY RACE AND COMPOUND")
print(
    slopes.assign(
        Negative=slopes["SlopeSecondsPerLap"] < 0
    )
    .groupby(["Country", "Compound"])
    .agg(
        Stints=("Negative", "size"),
        NegativeStints=("Negative", "sum"),
        MedianSlope=("SlopeSecondsPerLap", "median"),
        MedianSamples=("Samples", "median"),
    )
    .round(4)
    .to_string()
)

# Examine the most negative individual stint estimates.
print("\n2. TEN MOST NEGATIVE STINTS")
worst = slopes.nsmallest(10, "SlopeSecondsPerLap")
print(
    worst[
        [
            "Country", "DriverNumber", "StintNumber",
            "Compound", "Samples", "UniqueTyreAges",
            "SlopeSecondsPerLap",
        ]
    ].round(4).to_string(index=False)
)

# Inspect the underlying laps for those stints.
print("\n3. UNDERLYING LAPS FOR THE FIVE MOST NEGATIVE STINTS")

for row in worst.head(5).itertuples(index=False):
    group = laps[
        (laps["Country"] == row.Country)
        & (laps["DriverNumber"] == row.DriverNumber)
        & (laps["StintNumber"] == row.StintNumber)
        & (laps["Compound"] == row.Compound)
    ].sort_values("LapNumber")

    print(
        f"\n{row.Country} | Driver {row.DriverNumber} | "
        f"Stint {row.StintNumber} | {row.Compound} | "
        f"Estimated slope {row.SlopeSecondsPerLap:.4f}"
    )

    columns = [
        "LapNumber", "LapTimeSeconds",
        "TyreLife", "IsPitOutLap",
    ]
    columns = [c for c in columns if c in group.columns]

    print(group[columns].head(25).to_string(index=False))
