
import fastf1
import traceback
from pathlib import Path

fastf1.Cache.enable_cache(str(Path("data/cache").resolve()))

import requests

url = (
    "https://livetiming.formula1.com/static/"
    "2024/2024-07-07_British_Grand_Prix/"
    "2024-07-07_Race/SessionInfo.jsonStream"
)

try:
    response = requests.get(
        url,
        headers={"User-Agent": "BestHTTP"},
        timeout=15
    )
    print("F1 timing HTTP status:", response.status_code)
    print("Response preview:", response.text[:200])
except requests.RequestException as exc:
    print("F1 timing request failed:", repr(exc))


try:
    session = fastf1.get_session(
        2024, "British Grand Prix", "R"
    )

    print("Session created:", session)

    session.load(
        laps=True,
        telemetry=False,
        weather=False,
        messages=False,
    )

    print("\nSession load completed.")

    laps = session.laps
    print("Lap data loaded:", not laps.empty)
    print("Lap rows:", len(laps))
    print("Drivers:", laps["Driver"].nunique())

except Exception:
    traceback.print_exc()
