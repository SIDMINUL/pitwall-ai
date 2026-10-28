
from pathlib import Path

from pitwall.analytics.race_planner import evaluate_strategy_robustness
from pitwall.data.jolpica_loader import (
    find_round_number,
    get_event_names,
    load_recent_laps,
)
from pitwall.data.circuits import get_race_by_event


TARGET_YEAR = 2026
TARGET_EVENT = "United States Grand Prix"
REFERENCE_YEAR = 2025

# Resolve the target event from the API instead of hardcoding its round.
events = get_event_names(TARGET_YEAR)
target_round = find_round_number(TARGET_YEAR, TARGET_EVENT)

if target_round is None:
    raise RuntimeError(
        f"Could not find {TARGET_EVENT} in the {TARGET_YEAR} event list."
    )

print(f"Target event: {TARGET_EVENT}")
print(f"Target year: {TARGET_YEAR}")
print(f"Target round: {target_round}")

# Obtain the configured race length for the target circuit.
race_config = get_race_by_event(TARGET_EVENT)

if not race_config:
    raise RuntimeError(
        f"No circuit configuration found for {TARGET_EVENT}."
    )

total_laps = int(race_config["total_laps"])

# Use the latest three completed rounds of the reference season.
reference_events = get_event_names(REFERENCE_YEAR)
reference_rounds = list(range(max(1, len(reference_events) - 2),
                              len(reference_events) + 1))

print(f"Historical reference year: {REFERENCE_YEAR}")
print(f"Historical reference rounds: {reference_rounds}")
print(f"Target race distance: {total_laps} laps")

laps = load_recent_laps(
    year=REFERENCE_YEAR,
    rounds=reference_rounds,
)

if laps.empty:
    raise RuntimeError("No historical lap data was loaded.")

result = evaluate_strategy_robustness(
    laps,
    total_laps=total_laps,
    pit_loss_values=(18.0, 23.0, 28.0),
    degradation_multipliers=(0.75, 1.0, 1.25),
    max_stops=3,
    top_n=10,
)

if result.empty:
    raise RuntimeError("No strategy results were generated.")

# Attach explicit target metadata to every row.
result.insert(0, "TargetEvent", TARGET_EVENT)
result.insert(1, "TargetYear", TARGET_YEAR)
result.insert(2, "TargetRound", target_round)
result.insert(3, "ReferenceYear", REFERENCE_YEAR)

output = Path("data/openf1/us_gp_2026_strategy_robustness.csv")
output.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(output, index=False)

print("\nTop strategy results:")
print(result.head(20).to_string(index=False))
print(f"\nSaved: {output}")
