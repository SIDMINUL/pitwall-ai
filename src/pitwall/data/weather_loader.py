
from __future__ import annotations

import requests
import pandas as pd

WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"


def load_historical_weather(
    date: str,
    latitude: float = 52.0786,
    longitude: float = -1.01694,
) -> pd.DataFrame:
    """
    Fetch historical hourly weather near Silverstone.

    The returned data is reanalysis weather, not direct
    measurements from the circuit's weather station.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date,
        "end_date": date,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "rain",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "cloud_cover",
            "pressure_msl",
        ]),
        "timezone": "Europe/London",
        "wind_speed_unit": "kmh",
    }

    response = requests.get(
        WEATHER_URL,
        params=params,
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    hourly = payload.get("hourly", {})

    if not hourly or "time" not in hourly:
        return pd.DataFrame()

    weather = pd.DataFrame(hourly)

    weather = weather.rename(columns={
        "time": "Time",
        "temperature_2m": "AirTemp",
        "relative_humidity_2m": "Humidity",
        "precipitation": "Precipitation",
        "rain": "Rainfall",
        "wind_speed_10m": "WindSpeed",
        "wind_direction_10m": "WindDirection",
        "wind_gusts_10m": "WindGusts",
        "cloud_cover": "CloudCover",
        "pressure_msl": "Pressure",
    })

    weather["Time"] = pd.to_datetime(weather["Time"])
    weather["WeatherSource"] = "Open-Meteo historical reanalysis"

    return weather


FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def load_race_weather_forecast(
    race_date: str,
    latitude: float,
    longitude: float,
    timezone: str = "auto",
) -> pd.DataFrame:
    """
    Fetch hourly forecast weather for a race date.

    Forecast availability depends on the API's current forecast
    horizon. Returned weather is modelled forecast data near the
    circuit, not direct trackside measurements.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation_probability",
            "precipitation",
            "rain",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "cloud_cover",
            "weather_code",
        ]),
        "start_date": race_date,
        "end_date": race_date,
        "timezone": timezone,
        "wind_speed_unit": "kmh",
    }

    response = requests.get(
        FORECAST_URL,
        params=params,
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    hourly = payload.get("hourly", {})

    if not hourly or "time" not in hourly:
        return pd.DataFrame()

    weather = pd.DataFrame(hourly)

    weather = weather.rename(columns={
        "time": "Time",
        "temperature_2m": "AirTemp",
        "relative_humidity_2m": "Humidity",
        "precipitation_probability": "RainProbability",
        "precipitation": "Precipitation",
        "rain": "Rainfall",
        "wind_speed_10m": "WindSpeed",
        "wind_direction_10m": "WindDirection",
        "wind_gusts_10m": "WindGusts",
        "cloud_cover": "CloudCover",
        "weather_code": "WeatherCode",
    })

    weather["Time"] = pd.to_datetime(weather["Time"])
    weather["WeatherSource"] = "Open-Meteo forecast"

    return weather


def summarize_race_weather(
    weather: pd.DataFrame,
) -> dict:
    """Create a compact summary of the available hourly forecast."""
    if weather.empty:
        return {
            "status": "unavailable",
            "message": "No forecast data returned.",
        }

    result = {
        "status": "available",
        "hours_returned": int(len(weather)),
        "source": str(weather["WeatherSource"].iloc[0]),
    }

    for column in [
        "AirTemp",
        "RainProbability",
        "Precipitation",
        "WindSpeed",
        "WindGusts",
    ]:
        if column in weather.columns:
            values = pd.to_numeric(weather[column], errors="coerce")
            if values.notna().any():
                result[f"{column}_min"] = round(
                    float(values.min()), 1
                )
                result[f"{column}_max"] = round(
                    float(values.max()), 1
                )

    return result


def classify_race_weather(
    weather: pd.DataFrame,
    start_hour: int = 13,
    end_hour: int = 17,
) -> dict:
    """
    Summarize weather during an approximate local race window.

    The default window is only a fallback. Replace it with the
    actual race start time from the official weekend schedule.
    """
    if weather.empty:
        return {
            "scenario": "unknown",
            "confidence": "low",
            "reason": "No forecast data available.",
        }

    data = weather.copy()
    data["Time"] = pd.to_datetime(data["Time"])

    data = data[
        (data["Time"].dt.hour >= start_hour)
        & (data["Time"].dt.hour <= end_hour)
    ]

    if data.empty:
        return {
            "scenario": "unknown",
            "confidence": "low",
            "reason": "No forecast data for the selected race window.",
        }

    rain_probability = pd.to_numeric(
        data.get("RainProbability", pd.Series(dtype=float)),
        errors="coerce",
    ).max()

    precipitation = pd.to_numeric(
        data.get("Precipitation", pd.Series(dtype=float)),
        errors="coerce",
    ).max()

    if pd.notna(precipitation) and precipitation >= 1.0:
        scenario = "wet_possible"
        reason = "Forecast precipitation reaches at least 1 mm/hour."
    elif pd.notna(rain_probability) and rain_probability >= 40:
        scenario = "rain_risk"
        reason = "Rain probability reaches at least 40%."
    elif pd.notna(rain_probability) and rain_probability <= 10:
        scenario = "dry_likely"
        reason = "Forecast rain probability is low in this time window."
    else:
        scenario = "mixed_or_uncertain"
        reason = "Forecast does not clearly indicate dry or wet conditions."

    return {
        "scenario": scenario,
        "confidence": "provisional",
        "reason": reason,
        "hours_checked": int(len(data)),
        "max_rain_probability": (
            float(rain_probability)
            if pd.notna(rain_probability)
            else None
        ),
        "max_precipitation_mm": (
            float(precipitation)
            if pd.notna(precipitation)
            else None
        ),
        "max_wind_gust_kmh": (
            float(pd.to_numeric(
                data.get("WindGusts", pd.Series(dtype=float)),
                errors="coerce",
            ).max())
            if pd.to_numeric(
                data.get("WindGusts", pd.Series(dtype=float)),
                errors="coerce",
            ).notna().any()
            else None
        ),
    }

