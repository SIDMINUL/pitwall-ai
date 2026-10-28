
from __future__ import annotations

import time
import requests
import pandas as pd

BASE_URL = "https://api.jolpi.ca/ergast/f1"


def _time_to_seconds(value: str | None) -> float | None:
    if not value:
        return None

    try:
        parts = value.split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(value)
    except (ValueError, TypeError):
        return None


def _get_json(
    session: requests.Session,
    url: str,
    params: dict,
    max_retries: int = 5,
) -> dict:
    """Request JSON with backoff when the API rate-limits us."""
    for attempt in range(max_retries + 1):
        try:
            response = session.get(
                url,
                params=params,
                timeout=30,
            )

            if response.status_code == 429:
                if attempt == max_retries:
                    response.raise_for_status()

                retry_after = response.headers.get("Retry-After")
                try:
                    wait = float(retry_after)
                except (TypeError, ValueError):
                    wait = min(3 * (2 ** attempt), 60)

                print(f"API rate limit reached. Waiting {wait:.0f}s...")
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.json()

        except requests.RequestException:
            if attempt == max_retries:
                raise

            wait = min(2 * (2 ** attempt), 30)
            print(f"Request failed. Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError("Unable to retrieve data from Jolpica.")


def load_laps(year: int, round_number: int) -> pd.DataFrame:
    """Load lap times with correct timing-record pagination."""
    url = f"{BASE_URL}/{year}/{round_number}/laps.json"

    rows = []
    offset = 0
    limit = 100
    total = None

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "PitWallAI/1.0 (historical race analytics)"
        })

        while total is None or offset < total:
            payload = _get_json(
                session,
                url,
                {"limit": limit, "offset": offset},
            )["MRData"]

            total = int(payload.get("total", 0))
            races = payload["RaceTable"].get("Races", [])

            if not races:
                break

            returned_timings = 0

            for race in races:
                actual_year = int(race.get("season", year))
                actual_round = int(race.get("round", round_number))

                if actual_year != year or actual_round != round_number:
                    raise ValueError(
                        f"Requested {year} round {round_number}, "
                        f"but API returned {actual_year} round {actual_round}."
                    )

                for lap in race.get("Laps", []):
                    for timing in lap.get("Timings", []):
                        returned_timings += 1

                        rows.append({
                            "Year": actual_year,
                            "Round": actual_round,
                            "Race": race.get("raceName"),
                            "Circuit": race.get("Circuit", {}).get(
                                "circuitName"
                            ),
                            "LapNumber": int(lap["number"]),
                            "Driver": timing.get("driverId"),
                            "DriverId": timing.get("driverId"),
                            "LapTime": _time_to_seconds(
                                timing.get("time")
                            ),
                            "Position": (
                                int(timing["position"])
                                if timing.get("position")
                                else None
                            ),
                        })

            if returned_timings == 0:
                break

            # Jolpica offset counts timing records, not lap objects.
            offset += returned_timings

            if offset < total:
                time.sleep(0.3)

    columns = [
        "Year", "Round", "Race", "Circuit", "LapNumber",
        "Driver", "DriverId", "LapTime", "Position",
    ]

    df = pd.DataFrame(rows, columns=columns)

    if not df.empty:
        df["Position"] = df["Position"].astype("Int64")
        df["LapNumber"] = df["LapNumber"].astype("int64")

    return df



def find_round_number(year: int, event_name: str) -> int:
    """Find a race round by matching its name or circuit."""
    url = f"{BASE_URL}/{year}/races.json"

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "PitWallAI/1.0 (historical race analytics)"
        })

        payload = _get_json(
            session,
            url,
            {"limit": 100},
        )["MRData"]

    races = payload["RaceTable"].get("Races", [])
    query = event_name.strip().casefold()

    if not query:
        raise ValueError("Enter a Grand Prix or circuit name.")

    # Prefer a race-name match, e.g. "British Grand Prix".
    exact = [
        race for race in races
        if race.get("raceName", "").casefold() == query
    ]
    if exact:
        return int(exact[0]["round"])

    matches = [
        race for race in races
        if query in race.get("raceName", "").casefold()
        or query in race.get("Circuit", {}).get(
            "circuitName", ""
        ).casefold()
    ]

    if len(matches) == 1:
        return int(matches[0]["round"])

    if len(matches) > 1:
        names = ", ".join(
            f'{race["raceName"]} (round {race["round"]})'
            for race in matches
        )
        raise ValueError(f"Event name is ambiguous: {names}")

    raise ValueError(
        f"No race matching '{event_name}' found in {year}."
    )


def get_event_names(year: int) -> list[str]:
    """Return all Grand Prix names listed for a season."""
    url = f"{BASE_URL}/{year}/races.json"

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "PitWallAI/1.0 (historical race analytics)"
        })

        payload = _get_json(
            session,
            url,
            {"limit": 100},
        )["MRData"]

    races = payload["RaceTable"].get("Races", [])

    return [
        race["raceName"]
        for race in sorted(
            races,
            key=lambda race: int(race["round"]),
        )
    ]


def load_recent_laps(
    year: int,
    rounds: list[int],
    delay_seconds: float = 2.0,
) -> pd.DataFrame:
    """Load lap records for requested rounds and verify round identity."""
    import time

    frames = []

    for round_number in rounds:
        try:
            print(f"Loading {year}, round {round_number}...")
            laps = load_laps(year, round_number)

            if laps.empty:
                print(f"  No records returned for round {round_number}.")
                continue

            # Reject responses that do not match the requested race.
            returned_rounds = pd.to_numeric(
                laps["Round"], errors="coerce"
            ).dropna().unique()

            if len(returned_rounds) != 1 or int(returned_rounds[0]) != round_number:
                print(
                    f"  WARNING: Requested round {round_number}, "
                    f"but response contained rounds {returned_rounds.tolist()}. "
                    "Skipping response."
                )
                continue

            # Remove exact repeated records returned by the API.
            before = len(laps)
            laps = laps.drop_duplicates(
                subset=["Year", "Round", "DriverId", "LapNumber"],
                keep="first",
            ).copy()
            removed = before - len(laps)

            if removed:
                print(f"  Removed {removed} repeated driver-lap records.")

            frames.append(laps)
            print(
                f"  Accepted {len(laps)} rows for "
                f"{laps['Race'].iloc[0]}."
            )

        except Exception as exc:
            print(f"  Failed round {round_number}: {exc}")

        time.sleep(delay_seconds)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    combined = combined.drop_duplicates(
        subset=["Year", "Round", "DriverId", "LapNumber"],
        keep="first",
    ).reset_index(drop=True)

    return combined
