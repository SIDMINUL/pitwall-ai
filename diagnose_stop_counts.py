
from pitwall.analytics.pre_race import (
    StrategyConfig,
    generate_candidate_strategies,
)

config = StrategyConfig(
    total_laps=56,
    pit_loss_seconds=23.0,
    min_stint_laps=5,
    max_stops=3,
)

candidates = generate_candidate_strategies(
    driver_pace_seconds=102.0,
    config=config,
    pit_window_step=1,
)

if candidates.empty:
    print("No candidate strategies were generated.")
else:
    best = (
        candidates.sort_values("EstimatedRaceTimeSeconds")
        .groupby("Stops", as_index=False)
        .first()
        .sort_values("Stops")
    )

    print("\nBEST STRATEGY FOR EACH STOP COUNT")
    print(
        best[
            [
                "Stops",
                "CompoundSequence",
                "StintLengths",
                "EstimatedRaceTimeSeconds",
                "EstimatedPitLossSeconds",
            ]
        ].to_string(index=False)
    )

    one_stop = best.loc[
        best["Stops"] == 1, "EstimatedRaceTimeSeconds"
    ]

    if not one_stop.empty:
        baseline = float(one_stop.iloc[0])
        best["SecondsVsOneStop"] = (
            best["EstimatedRaceTimeSeconds"] - baseline
        ).round(2)

        print("\nTIME DIFFERENCE RELATIVE TO ONE STOP")
        print(
            best[["Stops", "SecondsVsOneStop"]]
            .to_string(index=False)
        )
