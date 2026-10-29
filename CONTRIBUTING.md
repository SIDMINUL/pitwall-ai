# Contributing to PitWall AI

Thanks for considering a contribution.

## Before opening a pull request

1. Keep changes focused and explain the motivation.
2. Do not commit API keys, credentials, private datasets, cache directories, or virtual environments.
3. Preserve existing interfaces unless a change is necessary and documented.
4. Clearly label estimated or simulated values; do not present unvalidated outputs as race predictions.
5. Include tests for new strategy calculations where practical.

## Local checks

From the repository root:

```bash
python -m compileall -q app.py src
python -m pytest -q
```

If pytest collection is unusually slow, report the observed command and timing rather than skipping or weakening tests.

## Pull request checklist

- [ ] The change has a clear description.
- [ ] Relevant tests or manual verification are documented.
- [ ] New dependencies are justified and included in dependency files.
- [ ] Documentation and limitations are updated.
- [ ] No secrets or unlicensed data are included.
