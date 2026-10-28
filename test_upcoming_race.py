
from datetime import date

from src.pitwall.data.circuits import get_upcoming_races
from src.pitwall.data.weather_loader import (
    load_race_weather_forecast,
    summarize_race_weather,
)
from src.pitwall.data.weather_loader import classify_race_weather

races = get_upcoming_races()

if not races:
    print("No upcoming races found in the configured calendar.")
    raise SystemExit(0)

print("UPCOMING RACES")
for race in races:
    print(
        f"{race['race_date']} | {race['event']} | "
        f"{race['circuit']} | {race['total_laps']} laps"
    )

# The first configured upcoming event.
race = races[0]

print(f"\nFetching forecast for {race['event']}...")
print("Race date:", race["race_date"])

try:
    weather = load_race_weather_forecast(
        race_date=race["race_date"],
        latitude=race["latitude"],
        longitude=race["longitude"],
    )

    print("\nWEATHER SUMMARY")
    print(summarize_race_weather(weather))
    print("\nRACE WEATHER CLASSIFICATION")
    print(classify_race_weather(weather))

    if not weather.empty:
        print("\nHOURLY FORECAST SAMPLE")
        print(weather.head(5).to_string(index=False))

except Exception as exc:
    print(f"Forecast unavailable: {exc}")
    print(
        "The strategy planner must continue to work without "
        "forecast data and mark weather as unknown."
    )
