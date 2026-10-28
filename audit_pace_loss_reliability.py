
from pathlib import Path

import pandas as pd

INPUT = Path("data/openf1/within_stint_pace_loss.csv")
OUTPUT = Path("data/openf1/pace_loss_reliability.csv")

df = pd.read_csv(INPUT)

stint_keys = [
    "Country",
    "SessionKey",
    "DriverNumber",
    "StintNumber",
    "Compound",
]

df["PaceLossSeconds"] = pd.to_numeric(
    df["MedianPaceLossSeconds"], errors="coerce"
)
df["TyreLife"] = pd.to_numeric(df["TyreLife"], errors="coerce")

# One record per independent stint and tyre age.
df = df.dropna(
    subset=stint_keys + ["TyreLife", "PaceLossSeconds"]
).drop_duplicates(subset=stint_keys + ["TyreLife"])

# Summarize the variation in observed pace loss at each tyre age.
by_age = (
    df.groupby(["Compound", "TyreLife"])
    .agg(
        MedianPaceLoss=("PaceLossSeconds", "median"),
        Q25=("PaceLossSeconds", lambda s: s.quantile(0.25)),
        Q75=("PaceLossSeconds", lambda s: s.quantile(0.75)),
    )
    .reset_index()
)


counts = (
    df.groupby(["Compound", "TyreLife"], as_index=False)
    .agg(
        UniqueStints=(
            "SessionKey",
            lambda _: 0,  # Temporary placeholder; replaced below.
        )
    )
)

# Compute counts explicitly to avoid ambiguity about grouped columns.
count_records = []

for (compound, tyre_age), group in df.groupby(
    ["Compound", "TyreLife"]
):
    unique_count = group[stint_keys].drop_duplicates().shape[0]

    count_records.append({
        "Compound": compound,
        "TyreLife": tyre_age,
        "UniqueStints": unique_count,
    })

counts = pd.DataFrame(count_records)


by_age = by_age.merge(
    counts,
    on=["Compound", "TyreLife"],
    how="left",
)

by_age["IQR"] = by_age["Q75"] - by_age["Q25"]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
by_age.to_csv(OUTPUT, index=False)

print(f"Saved: {OUTPUT}")

print("\nTyre-age estimates supported by at least 10 independent stints:")
supported = by_age[by_age["UniqueStints"] >= 10].sort_values(
    ["Compound", "TyreLife"]
)
print(supported.round(3).to_string(index=False))

print("\nSupport by compound:")
support_summary = (
    by_age.groupby("Compound")
    .agg(
        TyreAgeValues=("TyreLife", "nunique"),
        MedianStintsPerAge=("UniqueStints", "median"),
        MinStintsPerAge=("UniqueStints", "min"),
        MaxStintsPerAge=("UniqueStints", "max"),
    )
)
print(support_summary.to_string())
