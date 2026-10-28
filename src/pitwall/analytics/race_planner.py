from __future__ import annotations

from dataclasses import replace
from itertools import product

import pandas as pd

from pitwall.analytics.pre_race import (
    StrategyConfig,
    build_driver_pace_profile,
    generate_candidate_strategies,
    simulate_strategy,
)



def plan_race_strategies(
    historical_laps: pd.DataFrame,
    *,
    total_laps: int,
    pit_loss_seconds: float,
    max_stops: int = 3,
    top_n: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate baseline strategy candidates using explicit illustrative assumptions."""
    profile = build_driver_pace_profile(historical_laps)

    if profile.empty:
        return profile, pd.DataFrame()

    config = StrategyConfig(
        total_laps=total_laps,
        pit_loss_seconds=pit_loss_seconds,
        max_stops=max_stops,
    )

    all_rows = []
    for driver in profile.itertuples(index=False):
        candidates = generate_candidate_strategies(
            driver_pace_seconds=float(driver.MedianLapSeconds),
            config=config,
        )
        if candidates.empty:
            continue

        candidates = (
            candidates.sort_values("EstimatedRaceTimeSeconds")
            .groupby("Stops", group_keys=False)
            .head(top_n)
            .copy()
        )
        candidates.insert(0, "DriverId", driver.DriverId)
        candidates["PaceDeltaSeconds"] = round(float(driver.PaceDeltaSeconds), 3)
        candidates["HistoricalLapSamples"] = int(driver.ValidLaps)
        candidates["Scenario"] = "Dry baseline"
        candidates["ModelStatus"] = "Illustrative assumptions"
        all_rows.append(candidates)

    if not all_rows:
        return profile, pd.DataFrame()

    strategies = pd.concat(all_rows, ignore_index=True)
    strategies["RankWithinStopCount"] = (
        strategies.groupby(["DriverId", "Stops"])["EstimatedRaceTimeSeconds"]
        .rank(method="dense")
        .astype(int)
    )
    strategies = strategies.sort_values(
        ["DriverId", "Stops", "RankWithinStopCount", "EstimatedRaceTimeSeconds"]
    ).reset_index(drop=True)
    return profile, strategies


def run_sensitivity_analysis(
    historical_laps: pd.DataFrame,
    *,
    total_laps: int,
    pit_loss_values: tuple[float, ...] = (18.0, 23.0, 28.0),
    degradation_multipliers: tuple[float, ...] = (0.75, 1.0, 1.25),
    max_stops: int = 3,
) -> pd.DataFrame:
    """
    Compare the best strategy for each stop count under a grid of assumptions.

    Degradation multipliers scale the illustrative per-compound degradation rates.
    Results are scenario comparisons, not calibrated race-time predictions.
    """
    if not pit_loss_values or not degradation_multipliers:
        raise ValueError("Provide at least one pit-loss value and degradation multiplier.")
    if any(value < 0 for value in pit_loss_values):
        raise ValueError("Pit-loss values must be non-negative.")
    if any(value <= 0 for value in degradation_multipliers):
        raise ValueError("Degradation multipliers must be greater than zero.")

    profile = build_driver_pace_profile(historical_laps)
    if profile.empty:
        return pd.DataFrame()

    rows = []
    for pit_loss, degradation_multiplier in product(
        pit_loss_values, degradation_multipliers
    ):
        base_config = StrategyConfig(
            total_laps=total_laps,
            pit_loss_seconds=float(pit_loss),
            max_stops=max_stops,
        )
        scaled_degradation = tuple(
            (compound, rate * degradation_multiplier)
            for compound, rate in base_config.compound_degradation
        )
        config = replace(base_config, compound_degradation=scaled_degradation)
        scenario_name = (
            f"pit_loss_{pit_loss:g}s__degradation_{degradation_multiplier:.2f}x"
        )

        for driver in profile.itertuples(index=False):
            candidates = generate_candidate_strategies(
                driver_pace_seconds=float(driver.MedianLapSeconds),
                config=config,
            )
            if candidates.empty:
                continue

            best_by_stops = (
                candidates.sort_values("EstimatedRaceTimeSeconds")
                .groupby("Stops", as_index=False)
                .head(1)
            )
            for candidate in best_by_stops.itertuples(index=False):
                rows.append({
                    "DriverId": driver.DriverId,
                    "PaceDeltaSeconds": round(float(driver.PaceDeltaSeconds), 3),
                    "HistoricalLapSamples": int(driver.ValidLaps),
                    "Scenario": scenario_name,
                    "PitLossSeconds": float(pit_loss),
                    "DegradationMultiplier": float(degradation_multiplier),
                    "Stops": int(candidate.Stops),
                    "CompoundSequence": candidate.CompoundSequence,
                    "StintLengths": candidate.StintLengths,
                    "EstimatedRaceTimeSeconds": float(candidate.EstimatedRaceTimeSeconds),
                    "ModelStatus": "Sensitivity analysis; illustrative assumptions",
                })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["ScenarioRankForDriver"] = (
        result.groupby(["DriverId", "Scenario"])["EstimatedRaceTimeSeconds"]
        .rank(method="dense")
        .astype(int)
    )
    result["IsBestScenarioOption"] = result["ScenarioRankForDriver"].eq(1)
    return result.sort_values(
        ["DriverId", "Scenario", "EstimatedRaceTimeSeconds"]
    ).reset_index(drop=True)


def evaluate_strategy_robustness(
    historical_laps: pd.DataFrame,
    *,
    total_laps: int,
    pit_loss_values: tuple[float, ...] = (18.0, 23.0, 28.0),
    degradation_multipliers: tuple[float, ...] = (0.75, 1.0, 1.25),
    max_stops: int = 3,
    top_n: int = 10,
) -> pd.DataFrame:
    """Evaluate strategies across illustrative parameter scenarios.

    Tied best strategies share one fractional win credit per scenario.
    Regret measures the additional estimated time relative to the
    best candidate in the same scenario.

    Results describe robustness within the generated strategy universe,
    not real-world race-win probabilities or calibrated predictions.
    """
    import numpy as np

    if total_laps < 2:
        raise ValueError("total_laps must be at least 2.")
    if max_stops < 1:
        raise ValueError("max_stops must be at least 1.")
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")
    if not pit_loss_values or not degradation_multipliers:
        raise ValueError(
            "Provide at least one pit-loss value and degradation multiplier."
        )
    if not all(
        np.isfinite(value) and value >= 0
        for value in pit_loss_values
    ):
        raise ValueError("Pit-loss values must be finite and non-negative.")
    if not all(
        np.isfinite(value) and value > 0
        for value in degradation_multipliers
    ):
        raise ValueError(
            "Degradation multipliers must be finite and greater than zero."
        )

    profile = build_driver_pace_profile(historical_laps)
    if profile.empty:
        return pd.DataFrame()

    scenario_grid = list(product(pit_loss_values, degradation_multipliers))
    all_results = []

    for driver in profile.itertuples(index=False):
        base_config = StrategyConfig(
            total_laps=total_laps,
            pit_loss_seconds=23.0,
            max_stops=max_stops,
        )

        candidates = generate_candidate_strategies(
            driver_pace_seconds=float(driver.MedianLapSeconds),
            config=base_config,
            pit_window_step=2,
            retain_top_n=None,
        )

        if candidates.empty:
            continue

        strategy_keys = ["Stops", "CompoundSequence", "StintLengths"]
        strategies = candidates[strategy_keys].drop_duplicates()

        scenario_rows = []

        for pit_loss, multiplier in scenario_grid:
            config = StrategyConfig(
                total_laps=total_laps,
                pit_loss_seconds=float(pit_loss),
                max_stops=max_stops,
                compound_pace_delta=base_config.compound_pace_delta,
                compound_degradation=tuple(
                    (compound, rate * float(multiplier))
                    for compound, rate in base_config.compound_degradation
                ),
            )

            for strategy in strategies.itertuples(index=False):
                compounds = tuple(
                    part.strip()
                    for part in strategy.CompoundSequence.split(" -> ")
                )
                lengths = tuple(
                    int(part.strip())
                    for part in strategy.StintLengths.split("/")
                )

                simulated = simulate_strategy(
                    compounds,
                    lengths,
                    float(driver.MedianLapSeconds),
                    config,
                )

                scenario_rows.append({
                    "DriverId": driver.DriverId,
                    "PaceDeltaSeconds": float(driver.PaceDeltaSeconds),
                    "HistoricalLapSamples": int(driver.ValidLaps),
                    "Stops": int(simulated["Stops"]),
                    "CompoundSequence": simulated["CompoundSequence"],
                    "StintLengths": simulated["StintLengths"],
                    "PitLossSeconds": float(pit_loss),
                    "DegradationMultiplier": float(multiplier),
                    "Scenario": (
                        f"pit_loss_{pit_loss:g}s__degradation_{multiplier:.2f}x"
                    ),
                    "EstimatedRaceTimeSeconds": float(
                        simulated["EstimatedRaceTimeSeconds"]
                    ),
                })

        evaluated = pd.DataFrame(scenario_rows)
        if evaluated.empty:
            continue

        scenario_groups = evaluated.groupby("Scenario")
        evaluated["ScenarioRank"] = scenario_groups[
            "EstimatedRaceTimeSeconds"
        ].rank(method="min")

        best_time = scenario_groups[
            "EstimatedRaceTimeSeconds"
        ].transform("min")

        evaluated["RegretSeconds"] = (
            evaluated["EstimatedRaceTimeSeconds"] - best_time
        ).clip(lower=0.0)

        evaluated["IsScenarioWinner"] = np.isclose(
            evaluated["EstimatedRaceTimeSeconds"],
            best_time,
            rtol=0.0,
            atol=1e-6,
        )

        winner_count = evaluated.groupby("Scenario")[
            "IsScenarioWinner"
        ].transform("sum")

        # Each scenario contributes one win in total, shared equally
        # among strategies tied for the best estimated time.
        evaluated["FractionalWinCredit"] = (
            evaluated["IsScenarioWinner"].astype(float) / winner_count
        )

        summary = (
            evaluated.groupby(strategy_keys, as_index=False)
            .agg(
                ScenariosEvaluated=("Scenario", "nunique"),
                ScenariosTiedForBest=("IsScenarioWinner", "sum"),
                FractionalScenarioWins=("FractionalWinCredit", "sum"),
                MedianEstimatedRaceTimeSeconds=(
                    "EstimatedRaceTimeSeconds", "median"
                ),
                MeanEstimatedRaceTimeSeconds=(
                    "EstimatedRaceTimeSeconds", "mean"
                ),
                WorstCaseEstimatedRaceTimeSeconds=(
                    "EstimatedRaceTimeSeconds", "max"
                ),
                RaceTimeStdSeconds=("EstimatedRaceTimeSeconds", "std"),
                MeanScenarioRank=("ScenarioRank", "mean"),
                MeanRegretSeconds=("RegretSeconds", "mean"),
                WorstCaseRegretSeconds=("RegretSeconds", "max"),
            )
        )

        summary["RaceTimeStdSeconds"] = (
            summary["RaceTimeStdSeconds"].fillna(0.0)
        )
        summary["FractionalWinRate"] = (
            summary["FractionalScenarioWins"]
            / summary["ScenariosEvaluated"]
        )
        summary["ScenarioWinRate"] = (
            summary["ScenariosTiedForBest"]
            / summary["ScenariosEvaluated"]
        )

        summary.insert(0, "DriverId", driver.DriverId)
        summary["PaceDeltaSeconds"] = round(
            float(driver.PaceDeltaSeconds), 3
        )
        summary["HistoricalLapSamples"] = int(driver.ValidLaps)
        summary["ModelStatus"] = (
            "Illustrative scenario robustness; not a calibrated forecast"
        )

        all_results.append(summary)

    if not all_results:
        return pd.DataFrame()

    result = pd.concat(all_results, ignore_index=True)

    return (
        result.sort_values(
            [
                "DriverId",
                "FractionalWinRate",
                "MeanRegretSeconds",
                "MedianEstimatedRaceTimeSeconds",
            ],
            ascending=[True, False, True, True],
        )
        .groupby("DriverId", group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )
