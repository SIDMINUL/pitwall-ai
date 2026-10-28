
from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pandas as pd


def estimate_compound_models(laps: pd.DataFrame) -> pd.DataFrame:
    """Estimate compound pace and tyre-age degradation from lap data.

    The estimates are descriptive linear fits, not causal measurements
    of tyre degradation.
    """
    required = {"Compound", "TyreLife", "LapTimeSeconds"}
    if not required.issubset(laps.columns):
        return pd.DataFrame()

    data = laps.copy()
    data["TyreLife"] = pd.to_numeric(data["TyreLife"], errors="coerce")
    data["LapTimeSeconds"] = pd.to_numeric(
        data["LapTimeSeconds"], errors="coerce"
    )

    data = data.dropna(
        subset=["Compound", "TyreLife", "LapTimeSeconds"]
    )
    data = data[
        data["TyreLife"].between(1, 80)
        & data["LapTimeSeconds"].between(40, 200)
    ].copy()

    # Exclude pit-in and pit-out laps when those fields are available.
    for column in ("PitInTime", "PitOutTime"):
        if column in data.columns:
            data = data[data[column].isna()]

    if data.empty:
        return pd.DataFrame()

    # Aggregate repeated observations to reduce the influence of outliers.
    grouped = (
        data.groupby(["Compound", "TyreLife"], as_index=False)
        .agg(
            MedianLapSeconds=("LapTimeSeconds", "median"),
            Samples=("LapTimeSeconds", "size"),
        )
    )

    rows = []

    for compound, group in grouped.groupby("Compound"):
        group = group.sort_values("TyreLife").copy()

        if group["TyreLife"].nunique() < 3:
            continue

        x = group["TyreLife"].to_numpy(dtype=float)
        y = group["MedianLapSeconds"].to_numpy(dtype=float)
        weights = group["Samples"].to_numpy(dtype=float)

        # Weighted linear regression gives more weight to tyre-age groups
        # with more observations.
        slope, intercept = np.polyfit(x, y, 1, w=weights)

        if not np.isfinite(slope) or not np.isfinite(intercept):
            continue

        predicted = intercept + slope * x
        residuals = y - predicted
        weighted_mean = np.average(y, weights=weights)

        ss_res = np.sum(weights * residuals**2)
        ss_tot = np.sum(weights * (y - weighted_mean) ** 2)

        r_squared = (
            float(1 - ss_res / ss_tot)
            if ss_tot > 0
            else float("nan")
        )

        
        median_pace = float(
            np.median(
                data.loc[
                    data["Compound"] == compound,
                    "LapTimeSeconds",
                ]
            )
        )

        rows.append({
            "Compound": str(compound),
            "Intercept": float(intercept),
            "DegradationSecondsPerLap": float(
                np.clip(slope, -0.05, 0.15)
            ),
            "MaxObservedTyreAge": int(group["TyreLife"].max()),
            "AgeGroups": int(group["TyreLife"].nunique()),
            "Samples": int(group["Samples"].sum()),
            "FitR2": r_squared,
            "MedianLapSeconds": median_pace,
            "EstimatedLapAtTyreAge1": float(intercept + slope),
            "EstimatedLapAtTyreAge5": float(intercept + slope * 5),
            "EstimatedLapAtTyreAge10": float(intercept + slope * 10),
            "ModelStatus": (
                "Limited coverage"
                if group["TyreLife"].nunique() < 5
                else "Fitted"
            ),
        })


    return pd.DataFrame(rows)


def optimize_strategies(
    laps: pd.DataFrame,
    total_laps: int,
    pit_loss_seconds: float = 22.0,
    max_stops: int = 3,
    min_stint_laps: int = 5,
) -> tuple[pd.DataFrame, str]:
    """Search feasible strategies using observed compound tyre-age data.

    Outputs are simplified model estimates, not validated race predictions.
    """
    empty = pd.DataFrame()

    # Validate scalar inputs before performing any calculations.
    if isinstance(total_laps, bool) or not isinstance(
        total_laps, (int, np.integer)
    ):
        return empty, "Race distance must be a whole number of laps."

    if total_laps < 2:
        return empty, "Race distance must be at least 2 laps."

    if isinstance(min_stint_laps, bool) or not isinstance(
        min_stint_laps, (int, np.integer)
    ):
        return empty, "Minimum stint length must be a whole number."

    if min_stint_laps < 1:
        return empty, "Minimum stint length must be at least 1 lap."

    if isinstance(max_stops, bool) or not isinstance(
        max_stops, (int, np.integer)
    ):
        return empty, "Maximum pit stops must be a whole number."

    if max_stops < 1:
        return empty, "Maximum pit stops must be at least 1."

    if not np.isfinite(pit_loss_seconds) or pit_loss_seconds < 0:
        return empty, "Pit-stop time loss must be finite and nonnegative."

    if total_laps < 2 * min_stint_laps:
        return empty, (
            "Race distance is too short for at least two feasible stints."
        )

    models = estimate_compound_models(laps)

    if len(models) < 2:
        return empty, (
            "Usable tyre-age data for at least two compounds is required. "
            "Load a session with compound, tyre-age, and lap-time data."
        )

    model_map = models.set_index("Compound").to_dict("index")
    compounds = sorted(model_map)

    max_stops = min(
        max_stops,
        3,
        total_laps // min_stint_laps - 1,
    )

    rows = []

    for stops in range(1, max_stops + 1):
        stint_count = stops + 1

        possible_cuts = range(
            min_stint_laps,
            total_laps - min_stint_laps + 1,
        )

        for cuts in combinations(possible_cuts, stops):
            boundaries = (0,) + cuts + (total_laps,)
            lengths = tuple(
                boundaries[i + 1] - boundaries[i]
                for i in range(stint_count)
            )

            if any(length < min_stint_laps for length in lengths):
                continue

            for sequence in product(compounds, repeat=stint_count):
                # Avoid consecutive stints on the same compound.
                if any(
                    sequence[i] == sequence[i + 1]
                    for i in range(stint_count - 1)
                ):
                    continue

                if len(set(sequence)) < 2:
                    continue

                # Do not extrapolate beyond observed tyre-age coverage.
                if any(
                    length > model_map[compound]["MaxObservedTyreAge"]
                    or length < 1
                    for compound, length in zip(sequence, lengths)
                ):
                    continue

                estimated_lap_time = 0.0

                
                baseline_running_time = 0.0
                degradation_time = 0.0

                for compound, length in zip(sequence, lengths):
                    model = model_map[compound]
                    intercept = float(model["Intercept"])
                    slope = float(
                        model["DegradationSecondsPerLap"]
                    )

                    # Baseline pace is the model's estimate at tyre age 1.
                    age_one_pace = intercept + slope
                    baseline_running_time += age_one_pace * length

                    # Only the change from age-one pace is degradation.
                    degradation_time += slope * sum(
                        age - 1 for age in range(1, length + 1)
                    )

                estimated_lap_time = (
                    baseline_running_time + degradation_time
                )


                pit_loss = stops * float(pit_loss_seconds)
                total_time = estimated_lap_time + pit_loss

                used_models = [
                    model_map[c] for c in set(sequence)
                ]

                fit_values = [
                    float(m["FitR2"])
                    for m in used_models
                    if pd.notna(m.get("FitR2"))
                    and np.isfinite(float(m["FitR2"]))
                ]

                rows.append({
                    "Stops": stops,
                    "Stints": stint_count,
                    "CompoundSequence": " -> ".join(sequence),
                    "StintLengths": " / ".join(map(str, lengths)),
                    "BaselineRunningTimeSeconds": round(
                    baseline_running_time, 2),
                    "DegradationTimeSeconds": round(
                    degradation_time, 2),
                    "EstimatedRaceTimeSeconds": round(total_time, 2),
                    "PitLossSeconds": round(pit_loss, 2),
                    "ModelAgeGroups": int(
                        sum(m["AgeGroups"] for m in used_models)
                    ),
                    "ModelSamples": int(
                        sum(m["Samples"] for m in used_models)
                    ),
                    "MeanModelFitR2": (
                        round(float(np.mean(fit_values)), 3)
                        if fit_values else None
                    ),
                })

    if not rows:
        return empty, (
            "No feasible strategy found. Check compound tyre-age coverage "
            "or adjust the race distance and minimum stint length."
        )

    results = pd.DataFrame(rows).sort_values(
        "EstimatedRaceTimeSeconds"
    ).reset_index(drop=True)

    results = results.drop_duplicates(
        subset=["Stops", "CompoundSequence", "StintLengths"]
    ).reset_index(drop=True)

    note = (
        "Strategies are ranked using compound-specific linear fits to "
        "observed lap times and tyre ages. Fit R2 measures descriptive "
        "fit quality, not causal tyre-degradation accuracy. Estimates "
        "exclude fuel-load effects, driver pace differences, traffic, "
        "safety cars, weather changes, track position, and tyre allocation "
        "rules. These are baseline simulations, not validated predictions."
    )

    return results, note


def analyze_strategy_robustness(
    strategies: pd.DataFrame,
    pit_loss_scenarios: tuple[float, ...] = (18.0, 22.0, 28.0),
    degradation_multipliers: tuple[float, ...] = (0.75, 1.0, 1.25),
) -> pd.DataFrame:
    """Rank strategies across pit-loss and degradation scenarios.

    This is a sensitivity analysis, not a probabilistic forecast.
    """
    required = {
        "Stops",
        "BaselineRunningTimeSeconds",
        "DegradationTimeSeconds",
    }

    if strategies.empty or not required.issubset(strategies.columns):
        return pd.DataFrame()

    if not pit_loss_scenarios or not degradation_multipliers:
        raise ValueError("At least one scenario of each type is required.")

    if any(
        not np.isfinite(x) or x < 0
        for x in pit_loss_scenarios
    ):
        raise ValueError("Pit-loss values must be finite and nonnegative.")

    if any(
        not np.isfinite(x) or x <= 0
        for x in degradation_multipliers
    ):
        raise ValueError(
            "Degradation multipliers must be finite and positive."
        )

    baseline = pd.to_numeric(
        strategies["BaselineRunningTimeSeconds"], errors="coerce"
    ).to_numpy(dtype=float)

    degradation = pd.to_numeric(
        strategies["DegradationTimeSeconds"], errors="coerce"
    ).to_numpy(dtype=float)

    stops = pd.to_numeric(
        strategies["Stops"], errors="coerce"
    ).to_numpy(dtype=float)

    if (
        not np.isfinite(baseline).all()
        or not np.isfinite(degradation).all()
        or not np.isfinite(stops).all()
        or (baseline <= 0).any()
        or (stops < 1).any()
    ):
        raise ValueError("Strategy components contain invalid values.")

    scenario_times = []

    for pit_loss in pit_loss_scenarios:
        for multiplier in degradation_multipliers:
            scenario_total = (
                baseline
                + degradation * multiplier
                + stops * pit_loss
            )
            scenario_times.append(scenario_total)

    scenario_matrix = np.vstack(scenario_times)

    # Rank each strategy within each scenario: 1 means fastest.
    ranks = (
        pd.DataFrame(scenario_matrix)
        .rank(axis=1, method="min")
        .to_numpy(dtype=float)
    )

    results = strategies.copy()
    results["MeanScenarioRank"] = ranks.mean(axis=0)
    results["WorstCaseRank"] = ranks.max(axis=0)
    results["WorstCaseTimeSeconds"] = scenario_matrix.max(axis=0)
    results["ScenarioWinRate"] = (ranks == 1).mean(axis=0)

    return results.sort_values(
        ["MeanScenarioRank", "WorstCaseRank", "WorstCaseTimeSeconds"]
    ).reset_index(drop=True)
