# Development Notes

## Current focus

PitWall AI is an evolving race-strategy analysis prototype. Keep the UI, model assumptions, and limitations aligned.

## Before a release

- Test from a clean Python 3.11 virtual environment.
- Verify every import used by `app.py` and `src/` is represented in dependency files.
- Run `python -m compileall -q app.py src`.
- Run `python -m pytest -q` and record the result.
- Verify both data-rich and data-limited sessions produce clear UI messaging.
- Check that no local cache, downloaded race data, credentials, or personal paths are tracked.
- Validate model outputs on synthetic data before interpreting historical rankings.

## Strategy-model cautions

The compound-specific linear fit is descriptive. Lap-time data is observational and can include driver, fuel, traffic, weather, track evolution, and session effects. Model fit statistics do not establish causality or out-of-sample accuracy.

Sensitivity analysis compares specified assumptions only. It does not model the probability distribution of real race conditions.

## Known development issue

The local environment has shown very slow pytest collection despite test calls completing quickly. Investigate import-time work, collection hooks, and environment behavior before treating test duration as acceptable.
