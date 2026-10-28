
from dataclasses import replace

from pitwall.analytics.pre_race import (
    StrategyConfig,
    generate_candidate_strategies,
)

base_config = StrategyConfig(
    total_laps=56,
    pit_loss_seconds=23.0,
    min_stint_laps=5,
    max_stops=3,
)

multipliers = [
    2.00, 2.05, 2.10, 2.15, 2.20, 2.25,
    2.30, 2.35, 2.40, 2.45, 2.50,
]

print("\nDEGRADATION SENSITIVITY")
print("-" * 85)

for multiplier in multipliers:
    config = replace(
        base_config,
        compound_degradation=tuple(
            (compound, rate * multiplier)
            for compound, rate
            in base_config.compound_degradation
        ),
    )

    candidates = generate_candidate_strategies(
        driver_pace_seconds=102.0,
        config=config,
        pit_window_step=1,
    )

    if candidates.empty:
        print(f"{multiplier:>6.2f}x | No candidates")
        continue

    best = (
        candidates.sort_values("EstimatedRaceTimeSeconds")
        .groupby("Stops", as_index=False)
        .first()
        .sort_values("EstimatedRaceTimeSeconds")
    )

    winner = best.iloc[0]

    times = {
        int(row.Stops): float(row.EstimatedRaceTimeSeconds)
        for row in best.itertuples(index=False)
    }

    print(
        f"{multiplier:>6.2f}x | "
        f"Winner: {int(winner['Stops'])} stop(s) | "
        f"1-stop: {times.get(1, float('nan')):.2f}s | "
        f"2-stop: {times.get(2, float('nan')):.2f}s | "
        f"3-stop: {times.get(3, float('nan')):.2f}s"
    )
