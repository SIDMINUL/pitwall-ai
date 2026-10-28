
from pitwall.data.jolpica_loader import load_recent_laps

YEAR = 2026
RECENT_ROUNDS = [15,16,17]

laps = load_recent_laps(YEAR, RECENT_ROUNDS)

print("Total rows:", len(laps))
print("\nRows by round:")
print(
    laps.groupby(["Year", "Round", "Race"])
    .size()
    .to_string()
)

print("\nDuplicate driver-lap records:")
print(
    laps.duplicated(
        subset=["Year", "Round", "DriverId", "LapNumber"]
    ).sum()
)

print("\nUnique rounds:", sorted(laps["Round"].unique()))
print("\nRecords per driver:")
print(
    laps.groupby("DriverId")["LapTime"]
    .count()
    .sort_values(ascending=False)
    .to_string()
)
