from __future__ import annotations

from pathlib import Path

from pitwall.analytics.race_planner import run_sensitivity_analysis
from pitwall.data.jolpica_loader import load_recent_laps
from pitwall.data.circuits import get_race_by_event


def main() -> None:
    project_root = Path.cwd()
    race = get_race_by_event("United States Grand Prix")
    if not race:
        raise ValueError("United States Grand Prix was not found in circuits.py")

    rounds = [15, 16, 17]
    print(f"Loading historical laps for rounds {rounds}...")
    laps = load_recent_laps(2026, rounds)
    if laps.empty:
        raise RuntimeError("No historical lap data was loaded.")

    print(f"Loaded {len(laps):,} lap records.")
    print(
        f"Running sensitivity analysis for {race['event']} "
        f"({race['total_laps']} laps)..."
    )

    results = run_sensitivity_analysis(
        laps,
        total_laps=int(race["total_laps"]),
        pit_loss_values=(18.0, 23.0, 28.0),
        degradation_multipliers=(0.75, 1.0, 1.25),
        max_stops=3,
    )
    if results.empty:
        raise RuntimeError("Sensitivity analysis returned no candidates.")

    output_path = project_root / "sensitivity_analysis.csv"
    results.to_csv(output_path, index=False)

    best = results[results["IsBestScenarioOption"]].copy()
    print("\nBEST STOP-COUNT OPTION PER DRIVER AND SCENARIO")
    print(
        best[
            [
                "DriverId",
                "Scenario",
                "Stops",
                "CompoundSequence",
                "StintLengths",
                "EstimatedRaceTimeSeconds",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved {len(results):,} rows to {output_path}")
    print("Reminder: all degradation rates and pit-loss scenarios are illustrative.")


if __name__ == "__main__":
    main()
