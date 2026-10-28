
from src.pitwall.data.tyre_loader import load_pit_stops

stops = load_pit_stops(2024, 12)

print("Pit stops:", len(stops))
print(stops.head(20).to_string(index=False))

if not stops.empty:
    print("\nPit stops per driver:")
    print(
        stops.groupby("Driver")
        .size()
        .sort_values(ascending=False)
    )
