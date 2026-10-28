
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.openf1.org/v1"
OUTPUT_DIR = Path("data/openf1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update({"User-Agent": "PitWallAI/0.1 research project"})


def fetch(endpoint, params, retries=3):
    """Fetch an OpenF1 endpoint with basic retry handling."""
    for attempt in range(retries):
        try:
            response = session.get(
                f"{BASE_URL}/{endpoint}",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    # Locate the 2024 Bahrain Grand Prix race session.
    sessions = fetch(
        "sessions",
        {
            "year": 2024,
            "country_name": "Bahrain",
            "session_name": "Race",
        },
    )

    if not sessions:
        print("No matching session found.")
        return

    race = sessions[0]
    session_key = race["session_key"]

    print("Event:", race.get("meeting_name"))
    print("Session:", race.get("session_name"))
    print("Session key:", session_key)

    # Retrieve actual lap times and tyre stint metadata.
    laps = fetch("laps", {"session_key": session_key})
    stints = fetch("stints", {"session_key": session_key})

    lap_df = pd.DataFrame(laps)
    stint_df = pd.DataFrame(stints)

    print("\nLap records:", len(lap_df))
    print("Stint records:", len(stint_df))

    if lap_df.empty or stint_df.empty:
        print("Lap or stint data is empty; cannot build tyre dataset.")
        return

    print("\nLap columns:", list(lap_df.columns))
    print("Stint columns:", list(stint_df.columns))

    required_laps = {"driver_number", "lap_number", "lap_duration"}
    required_stints = {
        "driver_number",
        "lap_start",
        "lap_end",
        "compound",
    }

    if not required_laps.issubset(lap_df.columns):
        print("Required lap fields are missing.")
        return

    if not required_stints.issubset(stint_df.columns):
        print("Required stint fields are missing.")
        return

    # Associate each lap with the stint covering that lap.
    merged = lap_df.merge(
        stint_df,
        on=["driver_number", "session_key"],
        how="left",
        suffixes=("", "_stint"),
    )

    merged = merged[
        merged["lap_number"].between(
            merged["lap_start"],
            merged["lap_end"],
        )
    ].copy()

    # Estimate tyre age for each lap.
    if "tyre_age_at_start" in merged.columns:
        start_age = merged["tyre_age_at_start"].fillna(0)
        merged["TyreLife"] = (
            start_age
            + merged["lap_number"]
            - merged["lap_start"]
        )
    else:
        merged["TyreLife"] = (
            merged["lap_number"] - merged["lap_start"] + 1
        )

    merged["LapTimeSeconds"] = pd.to_numeric(
        merged["lap_duration"], errors="coerce"
    )

    merged = merged.dropna(
        subset=["compound", "TyreLife", "LapTimeSeconds"]
    )

    merged = merged[
        merged["LapTimeSeconds"].between(40, 200)
    ].copy()

    merged = merged.rename(
        columns={
            "driver_number": "DriverNumber",
            "lap_number": "LapNumber",
            "compound": "Compound",
        }
    )

    output = OUTPUT_DIR / "bahrain_2024_tyre_laps.csv"
    merged.to_csv(output, index=False)

    print("\nUsable tyre-associated laps:", len(merged))
    print("\nCompound counts:")
    print(merged["Compound"].value_counts().to_string())

    print("\nSample:")
    print(
        merged[
            [
                "DriverNumber",
                "LapNumber",
                "LapTimeSeconds",
                "Compound",
                "TyreLife",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nSaved dataset:", output.resolve())


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print("OpenF1 request failed:", exc)
    except (KeyError, ValueError, TypeError) as exc:
        print("Data processing failed:", type(exc).__name__, exc)
