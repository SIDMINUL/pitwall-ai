
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    """
    Explicit assumptions for an initial strategy simulation.

    These are placeholders until circuit-specific and tyre-specific
    measurements are loaded. They must not be presented as official
    Pirelli data or a validated race forecast.
    """
    total_laps: int
    pit_loss_seconds: float = 22.0
    min_stint_laps: int = 5
    max_stops: int = 3

    # Illustrative dry-compound assumptions, in seconds per lap.
    # Replace with estimates from reliable historical tyre telemetry.
    compound_pace_delta: tuple[tuple[str, float], ...] = (
        ("SOFT", 0.0),
        ("MEDIUM", 0.35),
        ("HARD", 0.65),
    )

    # Illustrative degradation rates in seconds per tyre-age lap.
    compound_degradation: tuple[tuple[str, float], ...] = (
        ("SOFT", 0.075),
        ("MEDIUM", 0.045),
        ("HARD", 0.030),
    )



def build_driver_pace_profile(
    laps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate driver pace relative to the field within each race.

    Raw lap times from different circuits are not directly comparable.
    This function computes driver-versus-field pace differences per race,
    then aggregates those differences across races.

    PaceDeltaSeconds:
        Negative = faster than the field median.
        Positive = slower than the field median.

    This does not fully remove fuel, traffic, tyre, weather, or safety-car
    effects. It is a relative historical pace indicator, not a forecast.
    """
    required = {"DriverId", "LapNumber", "LapTime"}
    if not required.issubset(laps.columns):
        raise ValueError(
            "Lap data must contain DriverId, LapNumber and LapTime."
        )

    data = laps.copy()
    data["LapTime"] = pd.to_numeric(data["LapTime"], errors="coerce")
    data["LapNumber"] = pd.to_numeric(
        data["LapNumber"], errors="coerce"
    )

    data = data.dropna(
        subset=["DriverId", "LapNumber", "LapTime"]
    )
    data = data[data["LapTime"].between(60, 180)]

    if data.empty:
        return pd.DataFrame()

    # Use the race identifier when available, so circuit pace differences
    # do not contaminate driver comparisons across events.
    if "Round" in data.columns:
        race_columns = ["Round"]
    elif "Race" in data.columns:
        race_columns = ["Race"]
    else:
        race_columns = []

    # If race identity is unavailable, we cannot normalize across events.
    if not race_columns:
        raise ValueError(
            "Lap data must include a Round or Race column to normalize "
            "pace across different circuits."
        )

    # Remove unusually slow laps within each driver and race.
    group_columns = race_columns + ["DriverId"]

    def robust_filter(group: pd.DataFrame) -> pd.DataFrame:
        lower = group["LapTime"].quantile(0.10)
        upper = group["LapTime"].quantile(0.90)
        return group[group["LapTime"].between(lower, upper)]

    filtered_groups = []

    for _, group in data.groupby(group_columns, sort=False):
        filtered_groups.append(robust_filter(group))

    data = (
        pd.concat(filtered_groups, ignore_index=True)
        if filtered_groups
        else data.iloc[0:0].copy()
    )

    if data.empty:
        return pd.DataFrame()

    # Each driver's median pace within each race.
    per_race = (
        data.groupby(race_columns + ["DriverId"], as_index=False)
        .agg(
            RaceMedianLapSeconds=("LapTime", "median"),
            ValidLaps=("LapTime", "size"),
            FirstLap=("LapNumber", "min"),
            LastLap=("LapNumber", "max"),
        )
    )

    # Normalize within each race, not across raw lap times from circuits
    # with different lap lengths and characteristics.
    per_race["RaceFieldMedianSeconds"] = (
        per_race.groupby(race_columns)["RaceMedianLapSeconds"]
        .transform("median")
    )

    per_race["RacePaceDeltaSeconds"] = (
        per_race["RaceMedianLapSeconds"]
        - per_race["RaceFieldMedianSeconds"]
    )

    # Aggregate normalized pace differences across the available races.
    profile = (
        per_race.groupby("DriverId", as_index=False)
        .agg(
            MedianLapSeconds=("RaceMedianLapSeconds", "median"),
            ValidLaps=("ValidLaps", "sum"),
            FirstLap=("FirstLap", "min"),
            LastLap=("LastLap", "max"),
            PaceDeltaSeconds=("RacePaceDeltaSeconds", "median"),
            RacesSampled=("RacePaceDeltaSeconds", "count"),
            PaceVariabilitySeconds=("RacePaceDeltaSeconds", "std"),
        )
    )

    profile["PaceVariabilitySeconds"] = (
        profile["PaceVariabilitySeconds"].fillna(0.0)
    )

    return profile.sort_values(
        "PaceDeltaSeconds"
    ).reset_index(drop=True)


    # Trim unusually slow laps using each driver's own distribution.
    def robust_filter(group: pd.DataFrame) -> pd.DataFrame:
        lower = group["LapTime"].quantile(0.10)
        upper = group["LapTime"].quantile(0.90)
        return group[group["LapTime"].between(lower, upper)]

        data = data.groupby("DriverId", group_keys=False).apply(
        robust_filter,
        include_groups=False,
    ).reset_index(drop=True)

    profile = (
        data.groupby("DriverId", as_index=False)
        .agg(
            MedianLapSeconds=("LapTime", "median"),
            ValidLaps=("LapTime", "size"),
            FirstLap=("LapNumber", "min"),
            LastLap=("LapNumber", "max"),
        )
    )

    field_median = profile["MedianLapSeconds"].median()
    profile["PaceDeltaSeconds"] = (
        profile["MedianLapSeconds"] - field_median
    )

    return profile.sort_values(
        "MedianLapSeconds"
    ).reset_index(drop=True)


def simulate_strategy(
    stint_compounds: tuple[str, ...],
    stint_lengths: tuple[int, ...],
    pace_seconds: float,
    config: StrategyConfig,
    *,
    weather_penalty_seconds: float = 0.0,
    safety_car_pit_discount: float = 0.0,
) -> dict:
    """
    Calculate a transparent baseline estimate for one strategy.

    weather_penalty_seconds is an explicitly supplied scenario penalty.
    safety_car_pit_discount is the assumed reduction in pit loss,
    between 0 and config.pit_loss_seconds.

    Neither parameter is inferred automatically in this initial version.
    """
    if len(stint_compounds) != len(stint_lengths):
        raise ValueError("Each stint needs one compound and one length.")

    if sum(stint_lengths) != config.total_laps:
        raise ValueError("Stint lengths must equal total race laps.")

    if any(length < config.min_stint_laps for length in stint_lengths):
        raise ValueError("A stint is shorter than the configured minimum.")

    pace_delta = dict(config.compound_pace_delta)
    degradation = dict(config.compound_degradation)

    unknown = set(stint_compounds) - set(pace_delta)
    if unknown:
        raise ValueError(f"Missing pace assumptions for: {sorted(unknown)}")

    unknown = set(stint_compounds) - set(degradation)
    if unknown:
        raise ValueError(
            f"Missing degradation assumptions for: {sorted(unknown)}"
        )

    green_flag_lap_time = 0.0
    stint_start_lap = 1

    for compound, length in zip(stint_compounds, stint_lengths):
        age = np.arange(1, length + 1, dtype=float)

        green_flag_lap_time += np.sum(
            pace_seconds
            + pace_delta[compound]
            + degradation[compound] * (age - 1)
        )

        stint_start_lap += length

    stops = len(stint_compounds) - 1
    discount = float(
        np.clip(safety_car_pit_discount, 0, config.pit_loss_seconds)
    )
    pit_loss = stops * config.pit_loss_seconds - discount

    estimated_time = (
        green_flag_lap_time
        + pit_loss
        + weather_penalty_seconds
    )

    return {
        "Stops": stops,
        "CompoundSequence": " -> ".join(stint_compounds),
        "StintLengths": " / ".join(map(str, stint_lengths)),
        "EstimatedRaceTimeSeconds": round(
            float(estimated_time), 2
        ),
        "EstimatedPitLossSeconds": round(float(pit_loss), 2),
        "WeatherScenarioPenaltySeconds": round(
            float(weather_penalty_seconds), 2
        ),
    }


def generate_candidate_strategies(
    driver_pace_seconds: float,
    config: StrategyConfig,
    *,
    weather_penalty_seconds: float = 0.0,
    safety_car_pit_discount: float = 0.0,
    pit_window_step: int = 2,
    retain_top_n: int | None = 10,
) -> pd.DataFrame:
    """
    Efficient candidate search using configurable pit-window steps.

    Compound assumptions remain illustrative until calibrated with
    reliable tyre data. This is a baseline simulator, not a validated
    race prediction.
    """
    if config.total_laps < 2 * config.min_stint_laps:
        return pd.DataFrame()

    compounds = [name for name, _ in config.compound_pace_delta]
    max_stops = min(
        config.max_stops,
        3,
        config.total_laps // config.min_stint_laps - 1,
    )

    if pit_window_step < 1:
        raise ValueError("pit_window_step must be at least 1.")

    rows = []

    for stops in range(1, max_stops + 1):
        stint_count = stops + 1

        # Sample pit windows instead of testing every lap.
        possible_cuts = list(
            range(
                config.min_stint_laps,
                config.total_laps - config.min_stint_laps + 1,
                pit_window_step,
            )
        )

        final_cut = config.total_laps - config.min_stint_laps
        if final_cut not in possible_cuts:
            possible_cuts.append(final_cut)

        for cuts in combinations(possible_cuts, stops):
            boundaries = (0,) + cuts + (config.total_laps,)
            lengths = tuple(
                boundaries[i + 1] - boundaries[i]
                for i in range(stint_count)
            )

            if any(
                length < config.min_stint_laps for length in lengths
            ):
                continue

            for sequence in product(compounds, repeat=stint_count):
                if any(
                    sequence[i] == sequence[i + 1]
                    for i in range(len(sequence) - 1)
                ):
                    continue

                if len(set(sequence)) < 2:
                    continue

                result = simulate_strategy(
                    sequence,
                    lengths,
                    driver_pace_seconds,
                    config,
                    weather_penalty_seconds=weather_penalty_seconds,
                    safety_car_pit_discount=safety_car_pit_discount,
                )

                rows.append(result)

    if not rows:
        return pd.DataFrame()

    results = pd.DataFrame(rows).drop_duplicates(
        subset=["Stops", "CompoundSequence", "StintLengths"]
    )

   
    # Optionally retain a shortlist for normal recommendations.
    # Robustness analysis can request all generated candidates.
    if retain_top_n is not None:
        if retain_top_n < 1:
            raise ValueError("retain_top_n must be at least 1.")

        results = (
            results.sort_values("EstimatedRaceTimeSeconds")
            .groupby("Stops", group_keys=False)
            .head(retain_top_n)
        )

    return results.sort_values(
        "EstimatedRaceTimeSeconds"
    ).reset_index(drop=True)




def recommend_for_each_driver(
    laps: pd.DataFrame,
    config: StrategyConfig,
    *,
    top_n: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produce candidate strategy rankings for every driver in a dataset.

    Returns:
        driver pace profile, ranked candidate strategy table.

    IMPORTANT:
    The lap dataset must be historical data used as a pace reference.
    The output is a baseline scenario simulation, not a validated
    prediction for an upcoming race.
    """
    profile = build_driver_pace_profile(laps)

    if profile.empty:
        return profile, pd.DataFrame()

    all_strategies = []

    for driver in profile.itertuples(index=False):
        candidates = generate_candidate_strategies(
            driver_pace_seconds=float(driver.MedianLapSeconds),
            config=config,
        )

        if candidates.empty:
            continue

        candidates.insert(0, "DriverId", driver.DriverId)
        candidates["DriverPaceDeltaSeconds"] = round(
            float(driver.PaceDeltaSeconds), 3
        )
        candidates["ValidHistoricalLaps"] = int(driver.ValidLaps)
        candidates["RankForDriver"] = np.arange(
            1, len(candidates) + 1
        )

        all_strategies.append(candidates.head(top_n))

    if not all_strategies:
        return profile, pd.DataFrame()

    return profile, pd.concat(
        all_strategies, ignore_index=True
    )
