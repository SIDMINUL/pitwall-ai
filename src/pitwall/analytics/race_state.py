
from __future__ import annotations

import pandas as pd


def _seconds(series: pd.Series) -> pd.Series:
    """Convert FastF1 timedeltas or Jolpica seconds to numeric seconds."""
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds()

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().any():
        return numeric

    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def _pace_table(laps: pd.DataFrame) -> pd.DataFrame:
    if laps.empty or "Driver" not in laps.columns:
        return pd.DataFrame()

    valid = laps.copy()
    valid["LapTimeSeconds"] = pd.to_numeric(
        valid["LapTimeSeconds"], errors="coerce"
    )

    
    # Keep only plausible lap times.
    valid = valid[
        valid["LapTimeSeconds"].between(40, 200, inclusive="both")
    ].copy()

    # FastF1 provides pit-in and pit-out information.
    # Jolpica lap records do not provide these columns.
    if "PitInTime" in valid.columns:
        valid = valid[valid["PitInTime"].isna()]

    if "PitOutTime" in valid.columns:
        valid = valid[valid["PitOutTime"].isna()]


    # Exclude pit/in-laps and out-laps when FastF1 supplies this field.
    if "PitInTime" in valid.columns:
        valid = valid[valid["PitInTime"].isna()]
    if "PitOutTime" in valid.columns:
        valid = valid[valid["PitOutTime"].isna()]

    if valid.empty:
        return pd.DataFrame()

    pace = (
        valid.groupby("Driver")["LapTimeSeconds"]
        .agg(
            ValidLaps="count",
            MedianLapSeconds="median",
            MeanLapSeconds="mean",
            StdDevSeconds="std",
            BestLapSeconds="min",
        )
        .reset_index()
    )

    pace["MedianGapToFastestSeconds"] = (
        pace["MedianLapSeconds"] - pace["MedianLapSeconds"].min()
    )

    numeric_columns = [
        "MedianLapSeconds", "MeanLapSeconds",
        "StdDevSeconds", "BestLapSeconds",
        "MedianGapToFastestSeconds",
    ]
    pace[numeric_columns] = pace[numeric_columns].round(3)

    return pace.sort_values("MedianLapSeconds").reset_index(drop=True)


def _build_from_jolpica(laps: pd.DataFrame) -> dict:
    data = laps.copy()

    data["LapTimeSeconds"] = pd.to_numeric(
        data.get("LapTime"), errors="coerce"
    )

    pace = _pace_table(data)

    results = pd.DataFrame()
    if not data.empty and "Position" in data.columns:
        results = (
            data.sort_values("LapNumber")
            .groupby("Driver", as_index=False)
            .tail(1)[["Driver", "LapNumber", "Position"]]
            .sort_values("Position", na_position="last")
            .reset_index(drop=True)
        )

    event_info = {}
    for column, key in [
        ("Year", "Season"),
        ("Round", "Round"),
        ("Race", "Event"),
        ("Circuit", "Circuit"),
    ]:
        if column in data.columns and not data.empty:
            event_info[key] = data[column].iloc[0]

    return {
        "event": event_info,
        "lap_count": int(len(data)),
        "valid_lap_count": int(data["LapTimeSeconds"].between(
            40, 200, inclusive="both"
        ).sum()),
        "driver_count": int(data["Driver"].nunique())
        if "Driver" in data.columns else 0,
        "driver_pace": pace,
        "tyre_stints": pd.DataFrame(),
        "weather_samples": pd.DataFrame(),
        "weather_summary": pd.DataFrame(),
        "results": results,
        "lap_data": data,
        "availability": {
            "Lap data": not data.empty,
            "Weather samples": False,
            "Session results": not results.empty,
            "Tyre compounds and stints": False,
            "Telemetry loaded": False,
            "Lap-by-lap positions": (
                "Position" in data.columns and not data.empty
            ),
        },
        "caveats": [
            "Lap times and positions are from the Jolpica historical API.",
            "This fallback does not provide tyre compounds, tyre age, weather or telemetry.",
            "Final recorded lap position is not necessarily the official finishing classification.",
            "Pit stops, traffic, fuel load, tyre condition and track evolution can affect lap times.",
            "The fastest or median lap is not a pure measure of driver skill or car capability.",
            "Only plausible lap times are included in the pace summary; this is not a full race-strategy model.",
            "Jolpica lap times are sanity-filtered, but pit laps and safety-car periods cannot be reliably excluded using the available fields.",
        ],
    }


def build_race_state(session) -> dict:
    # Jolpica fallback: a pandas DataFrame, not a FastF1 session.
    if isinstance(session, pd.DataFrame):
        return _build_from_jolpica(session)

    # FastF1 session path.
    try:
        laps = session.laps.copy()
    except Exception as exc:
        raise RuntimeError(
            "Lap data is unavailable from this FastF1 session."
        ) from exc

    try:
        weather = session.weather_data.copy()
    except Exception:
        weather = pd.DataFrame()

    try:
        results = session.results.copy()
    except Exception:
        results = pd.DataFrame()

    if "LapTime" in laps.columns:
        laps["LapTimeSeconds"] = _seconds(laps["LapTime"])
    else:
        laps["LapTimeSeconds"] = float("nan")

    pace = _pace_table(laps)

    stints = pd.DataFrame()
    stint_keys = [
        column for column in [
            "Driver", "Team", "Stint", "Compound", "FreshTyre"
        ] if column in laps.columns
    ]

    if stint_keys and "LapNumber" in laps.columns:
        aggregations = {"LapNumber": ["min", "max", "count"]}

        if "TyreLife" in laps.columns:
            aggregations["TyreLife"] = ["min", "max"]

        if "LapTimeSeconds" in laps.columns:
            aggregations["LapTimeSeconds"] = ["median", "min"]

        stints = laps.groupby(
            stint_keys, dropna=False
        ).agg(aggregations)

        stints.columns = [
            "_".join(str(value) for value in col if str(value))
            for col in stints.columns.to_flat_index()
        ]
        stints = stints.reset_index()

    weather_rows = []
    for column in [
        "AirTemp", "TrackTemp", "Humidity",
        "Pressure", "WindSpeed", "Rainfall",
    ]:
        if column in weather.columns:
            values = pd.to_numeric(
                weather[column], errors="coerce"
            ).dropna()

            if not values.empty:
                weather_rows.append({
                    "Metric": column,
                    "Min": round(float(values.min()), 2),
                    "Mean": round(float(values.mean()), 2),
                    "Max": round(float(values.max()), 2),
                })

    result_columns = [
        column for column in [
            "Abbreviation", "FullName", "TeamName",
            "GridPosition", "Position", "ClassifiedPosition",
            "Points", "Status",
        ] if column in results.columns
    ]
    result_table = results[result_columns].copy()

    event_info = {}
    event = getattr(session, "event", None)
    if event is not None:
        for key in ["EventName", "Location", "Country", "RoundNumber"]:
            try:
                value = event.get(key)
                if value is not None and not pd.isna(value):
                    event_info[key] = str(value)
            except Exception:
                pass

    return {
        "event": event_info,
        "lap_count": int(len(laps)),
        "valid_lap_count": int(
            pd.to_numeric(
                laps["LapTimeSeconds"], errors="coerce"
            ).between(40, 200, inclusive="both").sum()
        ),
        "driver_count": int(laps["Driver"].nunique())
        if "Driver" in laps.columns else 0,
        "driver_pace": pace,
        "tyre_stints": stints,
        "weather_samples": weather,
        "weather_summary": pd.DataFrame(weather_rows),
        "results": result_table,
        "lap_data": laps,
        "availability": {
            "Lap data": not laps.empty,
            "Weather samples": not weather.empty,
            "Session results": not results.empty,
            "Tyre compounds and stints": not stints.empty,
            "Telemetry loaded": False,
        },
        "caveats": [
            "Observed pace is not pure driver skill.",
            "Traffic, fuel, setup, tyre age and track evolution affect lap times.",
            "Weather samples are not a forecast.",
            "Team pace is only an imperfect proxy for car capability.",
            "Telemetry and predictive strategy simulation are not implemented yet.",
        ],
    }
