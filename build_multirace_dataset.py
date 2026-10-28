
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.openf1.org/v1"
YEAR = 2024

# Start with five races to validate the pipeline before scaling up.
TARGET_COUNTRIES = {
    "Bahrain",
    "Saudi Arabia",
    "Australia",
    "Japan",
    "China",
}

OUTPUT = Path("data/openf1/multirace_tyre_laps.csv")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

http = requests.Session()
http.headers.update({"User-Agent": "PitWallAI/0.1"})


def fetch(endpoint, params, retries=4):
    url = f"{BASE_URL}/{endpoint}"

    for attempt in range(retries):
        try:
            response = http.get(url, params=params, timeout=45)

            if response.status_code == 429:
                wait = 3 * (attempt + 1)
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            result = response.json()

            # Avoid sending requests too quickly.
            time.sleep(0.5)
            return result

        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Failed to fetch {endpoint}: {params}")


def main():
    sessions = fetch(
        "sessions",
        {"year": YEAR, "session_name": "Race"},
    )

    selected = [
        s for s in sessions
        if s.get("country_name") in TARGET_COUNTRIES
        and not s.get("is_cancelled", False)
    ]

    found = {s.get("country_name") for s in selected}
    missing = TARGET_COUNTRIES - found

    if missing:
        print("Warning: requested countries not found:", sorted(missing))

    if not selected:
        raise RuntimeError("No matching race sessions found.")

    all_races = []

    for session in selected:
        session_key = session["session_key"]
        country = session.get("country_name", "Unknown")
        circuit = session.get("circuit_short_name", "Unknown")

        print(f"\nLoading {YEAR} {country} GP | session {session_key}")

        try:
            laps = pd.DataFrame(
                fetch("laps", {"session_key": session_key})
            )
            stints = pd.DataFrame(
                fetch("stints", {"session_key": session_key})
            )
        except (requests.RequestException, RuntimeError) as exc:
            print("Failed to load race:", exc)
            continue

        if laps.empty or stints.empty:
            print("Skipping: lap or stint data is empty.")
            continue

        required_laps = {
            "session_key", "driver_number", "lap_number",
            "lap_duration", "is_pit_out_lap",
        }
        required_stints = {
            "session_key", "driver_number", "stint_number",
            "lap_start", "lap_end", "compound",
            "tyre_age_at_start",
        }

        if not required_laps.issubset(laps.columns):
            print("Skipping: missing lap fields.")
            continue

        if not required_stints.issubset(stints.columns):
            print("Skipping: missing stint fields.")
            continue

        # Match each lap to the stint for the same driver and session.
        merged = laps.merge(
            stints,
            on=["session_key", "driver_number"],
            how="inner",
            validate="many_to_many",
        )

        merged = merged[
            merged["lap_number"].between(
                merged["lap_start"],
                merged["lap_end"],
            )
        ].copy()

        # Each lap should map to exactly one stint.
        key = ["session_key", "driver_number", "lap_number"]
        duplicated = merged.duplicated(key, keep=False)

        if duplicated.any():
            print(
                "Warning: dropping ambiguous lap assignments:",
                int(duplicated.sum()),
            )
            merged = merged[~duplicated].copy()

        merged["TyreLife"] = (
            pd.to_numeric(
                merged["tyre_age_at_start"], errors="coerce"
            )
            + pd.to_numeric(
                merged["lap_number"], errors="coerce"
            )
            - pd.to_numeric(
                merged["lap_start"], errors="coerce"
            )
        )

        merged["LapTimeSeconds"] = pd.to_numeric(
            merged["lap_duration"], errors="coerce"
        )

        # Store session metadata and standardized field names.
        merged["Year"] = YEAR
        merged["Country"] = country
        merged["Circuit"] = circuit

        merged = merged.rename(
            columns={
                "session_key": "SessionKey",
                "driver_number": "DriverNumber",
                "lap_number": "LapNumber",
                "stint_number": "StintNumber",
                "compound": "Compound",
                "is_pit_out_lap": "IsPitOutLap",
                "duration_sector_1": "Sector1Seconds",
                "duration_sector_2": "Sector2Seconds",
                "duration_sector_3": "Sector3Seconds",
            }
        )

        keep = [
            "Year", "Country", "Circuit", "SessionKey",
            "DriverNumber", "LapNumber", "StintNumber",
            "Compound", "TyreLife", "LapTimeSeconds",
            "IsPitOutLap", "Sector1Seconds",
            "Sector2Seconds", "Sector3Seconds",
        ]

        for column in keep:
            if column not in merged.columns:
                merged[column] = pd.NA

        merged = merged[keep]
        all_races.append(merged)

        print(
            f"  Laps assigned to stints: {len(merged)} | "
            f"Drivers: {merged['DriverNumber'].nunique()} | "
            f"Compounds: {sorted(merged['Compound'].dropna().unique())}"
        )

    if not all_races:
        raise RuntimeError(
            "No usable race data collected. Check API access and logs."
        )

    data = pd.concat(all_races, ignore_index=True)

    data["LapNumber"] = pd.to_numeric(
        data["LapNumber"], errors="coerce"
    )
    data["LapTimeSeconds"] = pd.to_numeric(
        data["LapTimeSeconds"], errors="coerce"
    )
    data["TyreLife"] = pd.to_numeric(
        data["TyreLife"], errors="coerce"
    )

    # Remove invalid records, but preserve pit-out flags for later analysis.
    data = data.dropna(
        subset=[
            "SessionKey", "DriverNumber", "LapNumber",
            "LapTimeSeconds", "Compound", "TyreLife",
            "StintNumber",
        ]
    )

    data = data[
        data["LapTimeSeconds"].between(40, 200)
        & data["TyreLife"].between(0, 80)
    ].copy()

    data.to_csv(OUTPUT, index=False)

    print("\nMULTI-RACE DATASET SUMMARY")
    print("Rows:", len(data))
    print("Races:", data["SessionKey"].nunique())
    print("Countries:", sorted(data["Country"].unique()))
    print("Drivers:", data["DriverNumber"].nunique())
    print("Compounds:", sorted(data["Compound"].unique()))

    print("\nRows by race and compound:")
    print(
        data.groupby(["Country", "Compound"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    print("\nDuplicate driver/lap assignments:")
    print(
        data.duplicated(
            ["SessionKey", "DriverNumber", "LapNumber"]
        ).sum()
    )

    print("\nSaved:", OUTPUT.resolve())


if __name__ == "__main__":
    main()
