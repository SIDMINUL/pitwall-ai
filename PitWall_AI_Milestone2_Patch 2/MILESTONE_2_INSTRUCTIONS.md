# PitWall AI Milestone 2 patch

1. Back up your current `app.py`.
2. Extract this ZIP into `C:\my studies\projects\pitwall-ai`, allowing `app.py` to be overwritten.
3. Ensure `src\pitwall\analytics\race_state.py` is created.
4. Run PowerShell:

```powershell
cd "C:\my studies\projects\pitwall-ai"
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Load the same completed session that worked in Milestone 1. This milestone adds the unified race-state snapshot, driver pace, tyre stints, weather samples, results, and data-quality caveats. It does not yet make LLM recommendations.
