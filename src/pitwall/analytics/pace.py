import pandas as pd


def summarize_driver_pace(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Create a cautious baseline pace summary from valid lap times.

    This does not claim to isolate driver ability from car, traffic, tyres,
    fuel, setup, or changing track conditions.
    """
    required = {"Driver", "LapTime"}
    if laps is None or laps.empty or not required.issubset(laps.columns):
        return pd.DataFrame()

    data = laps[["Driver", "LapTime"]].copy()
    data["LapTimeSeconds"] = data["LapTime"].dt.total_seconds()
    data = data.dropna(subset=["Driver", "LapTimeSeconds"])
    data = data[(data["LapTimeSeconds"] > 0) & (data["LapTimeSeconds"] < 300)]

    if data.empty:
        return pd.DataFrame()

    summary = (
        data.groupby("Driver")["LapTimeSeconds"]
        .agg(
            ValidLaps="count",
            MedianLapSeconds="median",
            MeanLapSeconds="mean",
            StdDevSeconds="std",
            BestLapSeconds="min",
        )
        .reset_index()
    )
    summary["MedianLapSeconds"] = summary["MedianLapSeconds"].round(3)
    summary["MeanLapSeconds"] = summary["MeanLapSeconds"].round(3)
    summary["StdDevSeconds"] = summary["StdDevSeconds"].round(3)
    summary["BestLapSeconds"] = summary["BestLapSeconds"].round(3)
    return summary.sort_values("MedianLapSeconds").reset_index(drop=True)
