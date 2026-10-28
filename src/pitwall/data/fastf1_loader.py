
from pathlib import Path
import fastf1


def enable_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def load_session(year: int, event: str, session_name: str):
    session = fastf1.get_session(year, event, session_name)

    # Explicitly load all data needed by the race-state engine.
    session.load(
        laps=True,
        telemetry=False,
        weather=True,
        messages=False,
    )

    # Access results only after session.load() has completed.
    return session
