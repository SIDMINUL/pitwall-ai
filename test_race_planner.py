
import pandas as pd


from pitwall.data.jolpica_loader import load_recent_laps
from pitwall.analytics.race_planner import plan_race_strategies
from pitwall.data.circuits import get_race_by_event

YEAR = 2026
RECENT_ROUNDS = [15,16,17]
TARGET_EVENT = "United States Grand Prix"

race = get_race_by_event(TARGET_EVENT)

print(f"Loading {YEAR} laps from rounds {RECENT_ROUNDS}...")
laps = load_recent_laps(YEAR, RECENT_ROUNDS)

if laps.empty:
    raise RuntimeError("No recent lap data returned.")

print(f"Loaded {len(laps):,} lap records")
print(f"Drivers found: {laps['DriverId'].nunique()}")

print("\nTARGET RACE")
print(f"Event: {race['event']}")
print(f"Circuit: {race['circuit']}")
print(f"Date: {race['race_date']}")
print(f"Distance: {race['total_laps']} laps")
print(
    f"Pit loss: {race['pit_loss_seconds']} seconds "
    f"(estimated: {race['pit_loss_is_estimate']})"
)

pace, strategies = plan_race_strategies(
    laps,
    total_laps=race["total_laps"],
    pit_loss_seconds=race["pit_loss_seconds"],
    max_stops=3,
    top_n=3,
)

print("\nRECENT DRIVER PACE")
print(pace.to_string(index=False))

print("\nSTRATEGY CANDIDATES")
if strategies.empty:
    print("No strategies generated.")
else:
    columns = [
        "DriverId",
        "Stops",
        "RankWithinStopCount",
        "CompoundSequence",
        "StintLengths",
        "EstimatedRaceTimeSeconds",
        "PaceDeltaSeconds",
        "Scenario",
        "ModelStatus",
    ]

    print(strategies[columns].to_string(index=False))

    strategies.to_csv("race_strategy_candidates.csv", index=False)
    pace.to_csv("recent_driver_pace.csv", index=False)

    print("\nSaved race_strategy_candidates.csv")
    print("Saved recent_driver_pace.csv")
