
from src.pitwall.data.jolpica_loader import load_laps, find_round_number
from src.pitwall.analytics.pre_race import (
    StrategyConfig,
    recommend_for_each_driver,
)

YEAR = 2024
EVENT = "British Grand Prix"
TOTAL_LAPS = 52  # Example race distance; verify against loaded data.

print(f"Loading {YEAR} {EVENT}...")

round_number = find_round_number(YEAR, EVENT)
laps = load_laps(YEAR, round_number)

if laps.empty:
    raise RuntimeError("No historical lap data was returned.")

actual_lap_count = int(laps["LapNumber"].max())
print(f"Round: {round_number}")
print(f"Drivers: {laps['DriverId'].nunique()}")
print(f"Maximum lap number in data: {actual_lap_count}")

# Use the actual race distance from the loaded dataset.
config = StrategyConfig(total_laps=actual_lap_count)

profile, strategies = recommend_for_each_driver(
    laps,
    config,
    top_n=3,
)

print("\n=== Historical driver pace profile ===")
print(profile.to_string(index=False))

print("\n=== Top three illustrative strategies per driver ===")
if strategies.empty:
    print("No candidate strategies were generated.")
else:
    print(
        strategies[
            [
                "DriverId",
                "RankForDriver",
                "Stops",
                "CompoundSequence",
                "StintLengths",
                "EstimatedRaceTimeSeconds",
                "DriverPaceDeltaSeconds",
                "ValidHistoricalLaps",
            ]
        ].to_string(index=False)
    )

profile.to_csv("driver_pace_profile.csv", index=False)
strategies.to_csv("historical_strategy_candidates.csv", index=False)

print("\nSaved driver_pace_profile.csv")
print("Saved historical_strategy_candidates.csv")
print(
    "\nWARNING: These are illustrative simulations using assumed "
    "compound pace and degradation. They are not validated strategies."
)
