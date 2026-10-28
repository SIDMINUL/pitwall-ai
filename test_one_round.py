
from pitwall.data.jolpica_loader import load_laps

laps = load_laps(2026, 17)

print("Rows:", len(laps))
print("\nRace and round:")
print(laps[["Year", "Round", "Race", "Circuit"]].drop_duplicates().to_string(index=False))

print("\nDriver-lap duplicates:")
print(
    laps.duplicated(
        subset=["Year", "Round", "DriverId", "LapNumber"]
    ).sum()
)

print("\nRecords per driver:")
print(laps.groupby("DriverId")["LapNumber"].count().sort_values(ascending=False))
