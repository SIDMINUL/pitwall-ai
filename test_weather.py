
from src.pitwall.data.weather_loader import load_historical_weather

weather = load_historical_weather("2024-07-07")

print("Weather rows:", len(weather))
print(weather.head(10).to_string(index=False))

if not weather.empty:
    print("\nWeather summary:")
    print(
        weather[
            ["AirTemp", "Humidity", "Rainfall", "WindSpeed"]
        ].describe().round(2)
    )
