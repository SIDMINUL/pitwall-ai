
from __future__ import annotations

import requests
import pandas as pd

BASE_URL = "https://api.jolpi.ca/ergast/f1"


def load_pit_stops(year: int, round_number: int) -> pd.DataFrame:
    """Load historical pit-stop records from Jolpica."""
    url = (
        f"{BASE_URL}/{year}/{round_number}/pitstops.json"
    )

    rows = []
    offset = 0
    limit = 100

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "PitWallAI/1.0"
        })

        while True:
            response = session.get(
                url,
                params={"limit": limit, "offset": offset},
                timeout=30,
            )

            if response.status_code == 429:
                raise RuntimeError(
                    "Jolpica rate limit reached. Please retry later."
                )

            response.raise_for_status()
            payload = response.json()["MRData"]
            races = payload["RaceTable"].get("Races", [])

            if not races:
                break

            returned = 0

            for race in races:
                for stop in race.get("PitStops", []):
                    rows.append({
                        "Year": int(year),
                        "Round": int(round_number),
                        "Race": race.get("raceName"),
                        "Driver": stop.get("driverId"),
                        "Stop": int(stop["stop"]),
                        "LapNumber": int(stop["lap"]),
                        "PitDuration": stop.get("duration"),
                    })
                    returned += 1

            total = int(payload.get("total", 0))
            offset += returned

            if returned == 0 or offset >= total:
                break

    return pd.DataFrame(
        rows,
        columns=[
            "Year", "Round", "Race", "Driver",
            "Stop", "LapNumber", "PitDuration",
        ],
    )
