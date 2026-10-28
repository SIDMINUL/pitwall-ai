
from pathlib import Path

import fastf1

cache_dir = Path("data/fastf1_cache").resolve()
cache_dir.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(cache_dir))

print("FastF1 version:", fastf1.__version__)
print("Cache directory:", cache_dir)

# Use a completed historical race for the first connectivity test.
year = 2024
round_number = 1
session_type = "R"

try:
    session = fastf1.get_session(year, round_number, session_type)
    session.load(
        laps=True,
        telemetry=False,
        weather=False,
        messages=False,
    )

    laps = session.laps

    print("\nEvent:", session.event["EventName"])
    print("Lap records:", len(laps))
    print("Available columns:", list(laps.columns))

    required = {"Compound", "TyreLife", "LapTime"}
    missing = required - set(laps.columns)

    if missing:
        print("Missing tyre-data columns:", sorted(missing))
    else:
        tyre_data = laps[
            ["Driver", "LapNumber", "LapTime", "Compound", "TyreLife"]
        ].copy()

        tyre_data["LapTimeSeconds"] = (
            tyre_data["LapTime"].dt.total_seconds()
        )

        tyre_data = tyre_data.dropna(
            subset=["LapTimeSeconds", "Compound", "TyreLife"]
        )

        print("Rows with tyre data:", len(tyre_data))
        print("\nCompound counts:")
        print(tyre_data["Compound"].value_counts().to_string())

        print("\nSample records:")
        print(tyre_data.head(10).to_string(index=False))

        output = Path("data/fastf1_tyre_sample.csv")
        tyre_data.to_csv(output, index=False)
        print("\nSaved:", output.resolve())

except Exception as exc:
    print("\nFastF1 session load failed.")
    print("Error type:", type(exc).__name__)
    print("Error:", exc)
