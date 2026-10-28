
from src.pitwall.data.jolpica_loader import load_laps

laps = load_laps(2024, 12)

print("Rows:", len(laps))
print("Drivers:", laps["Driver"].nunique())
print("Laps:", laps["LapNumber"].nunique())
print("\nSample data:")
print(laps.head(10).to_string(index=False))

print("\nLap-time summary:")
print(
    laps.groupby("Driver")["LapTime"]
    .agg(["count", "mean", "min", "max"])
    .sort_values("mean")
    .head(10)
)
