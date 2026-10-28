
import pandas as pd
from pathlib import Path

path = Path("data/openf1/bahrain_2024_tyre_laps.csv")
df = pd.read_csv(path)

print("\nDATASET SHAPE")
print(df.shape)

print("\nMISSING VALUES")
print(
    df[["DriverNumber", "LapNumber", "LapTimeSeconds",
        "Compound", "TyreLife"]]
    .isna()
    .sum()
    .to_string()
)

print("\nDUPLICATE DRIVER/LAP ROWS")
print(
    df.duplicated(["DriverNumber", "LapNumber"]).sum()
)

print("\nTYRE AGE DISTRIBUTION")
print(df["TyreLife"].describe().to_string())

print("\nTYRE AGE COUNTS")
print(df["TyreLife"].value_counts().sort_index().head(20).to_string())

print("\nCOMPOUND AND TYRE AGE COVERAGE")
print(
    df.groupby("Compound").agg(
        Rows=("LapNumber", "size"),
        Drivers=("DriverNumber", "nunique"),
        MinAge=("TyreLife", "min"),
        MaxAge=("TyreLife", "max"),
        UniqueAges=("TyreLife", "nunique"),
        MedianLapTime=("LapTimeSeconds", "median"),
    ).to_string()
)

print("\nROWS WITH TYRE AGE ZERO")
print(
    df.loc[
        df["TyreLife"] <= 0,
        ["DriverNumber", "LapNumber", "Compound", "TyreLife"]
    ].head(20).to_string(index=False)
)

print("\nSAMPLE OF ONE DRIVER")
driver = df["DriverNumber"].value_counts().index[0]
print(
    df.loc[
        df["DriverNumber"] == driver,
        ["DriverNumber", "LapNumber", "LapTimeSeconds",
         "Compound", "TyreLife"]
    ].sort_values("LapNumber").head(20).to_string(index=False)
)
