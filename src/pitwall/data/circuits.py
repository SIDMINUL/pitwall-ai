
from __future__ import annotations

from datetime import date


# Upcoming 2026 races as of 18 October 2026.
# Coordinates are approximate circuit-area coordinates.
# Pit-loss values are initial assumptions, not measured circuit facts.

CIRCUITS = [
    {
        "event": "United States Grand Prix",
        "country": "United States",
        "circuit": "Circuit of The Americas",
        "race_date": "2026-10-25",
        "latitude": 30.1328,
        "longitude": -97.6411,
        "total_laps": 56,
        "pit_loss_seconds": 23.0,
        "pit_loss_is_estimate": True,
        "track_type": "Permanent circuit",
        "characteristics": "Mixed high-speed and technical corners",
    },
    {
        "event": "Mexico City Grand Prix",
        "country": "Mexico",
        "circuit": "Autódromo Hermanos Rodríguez",
        "race_date": "2026-11-01",
        "latitude": 19.4042,
        "longitude": -99.0907,
        "total_laps": 71,
        "pit_loss_seconds": 22.0,
        "pit_loss_is_estimate": True,
        "track_type": "Permanent circuit",
        "characteristics": "High altitude; traction and cooling matter",
    },
    {
        "event": "São Paulo Grand Prix",
        "country": "Brazil",
        "circuit": "Autódromo José Carlos Pace",
        "race_date": "2026-11-08",
        "latitude": -23.7036,
        "longitude": -46.6997,
        "total_laps": 71,
        "pit_loss_seconds": 22.0,
        "pit_loss_is_estimate": True,
        "track_type": "Permanent circuit",
        "characteristics": "Elevation changes; weather variability",
    },
    {
        "event": "Las Vegas Grand Prix",
        "country": "United States",
        "circuit": "Las Vegas Strip Circuit",
        "race_date": "2026-11-21",
        "latitude": 36.1147,
        "longitude": -115.1728,
        "total_laps": 50,
        "pit_loss_seconds": 22.0,
        "pit_loss_is_estimate": True,
        "track_type": "Street circuit",
        "characteristics": "Long straights; potentially cool conditions",
    },
    {
        "event": "Qatar Grand Prix",
        "country": "Qatar",
        "circuit": "Lusail International Circuit",
        "race_date": "2026-11-29",
        "latitude": 25.4900,
        "longitude": 51.4542,
        "total_laps": 57,
        "pit_loss_seconds": 22.0,
        "pit_loss_is_estimate": True,
        "track_type": "Permanent circuit",
        "characteristics": "High-speed corners; tyre-energy demands",
    },
    {
        "event": "Abu Dhabi Grand Prix",
        "country": "United Arab Emirates",
        "circuit": "Yas Marina Circuit",
        "race_date": "2026-12-06",
        "latitude": 24.4672,
        "longitude": 54.6031,
        "total_laps": 58,
        "pit_loss_seconds": 22.0,
        "pit_loss_is_estimate": True,
        "track_type": "Permanent circuit",
        "characteristics": "Mixed-speed layout; evening temperature changes",
    },
]


def get_upcoming_races(
    from_date: date | None = None,
) -> list[dict]:
    """Return races whose scheduled date has not passed."""
    reference_date = from_date or date.today()

    return [
        race.copy()
        for race in CIRCUITS
        if date.fromisoformat(race["race_date"]) >= reference_date
    ]


def get_race_by_event(event: str) -> dict:
    """Return circuit metadata for an exact event name."""
    for race in CIRCUITS:
        if race["event"] == event:
            return race.copy()

    raise ValueError(f"Unknown event: {event}")
