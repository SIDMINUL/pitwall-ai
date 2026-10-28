from pathlib import Path
import streamlit as st
import fastf1
import pandas as pd
import plotly.express as px
from src.pitwall.data.fastf1_loader import load_session, enable_cache
from src.pitwall.data.jolpica_loader import (
    load_laps,
    find_round_number,
    get_event_names,
)
from src.pitwall.data.circuits import (
    get_upcoming_races,
    get_race_by_event,
)
from src.pitwall.analytics.pre_race import (
    StrategyConfig,
    generate_candidate_strategies,
    simulate_strategy,
)
from src.pitwall.analytics.race_state import build_race_state
from src.pitwall.data.weather_loader import load_historical_weather
from src.pitwall.data.tyre_loader import load_pit_stops
from src.pitwall.analytics.strategy import (
    optimize_strategies,
    analyze_strategy_robustness,
)

def evaluate_candidate_robustness(
    candidates: pd.DataFrame,
    base_config: StrategyConfig,
    pit_loss_values=(18.0, 23.0, 28.0),
    degradation_multipliers=(0.75, 1.0, 1.25),
) -> pd.DataFrame:
    """Evaluate the same candidate strategies across sensitivity scenarios."""
    from dataclasses import replace
    from itertools import product
    import numpy as np

    if candidates.empty:
        return pd.DataFrame()

    # Compare a manageable shortlist, covering each stop count.
    shortlist = (
        candidates.sort_values("EstimatedRaceTimeSeconds")
        .groupby("Stops", group_keys=False)
        .head(10)
        .drop_duplicates(
            subset=["Stops", "CompoundSequence", "StintLengths"]
        )
        .reset_index(drop=True)
    )

    scenario_rows = []

    for pit_loss, multiplier in product(
        pit_loss_values, degradation_multipliers
    ):
        scenario_config = replace(
            base_config,
            pit_loss_seconds=float(pit_loss),
            compound_degradation=tuple(
                (compound, rate * multiplier)
                for compound, rate in base_config.compound_degradation
            ),
        )

        for candidate_id, candidate in shortlist.iterrows():
            compounds = tuple(
                part.strip()
                for part in candidate["CompoundSequence"].split("->")
            )
            lengths = tuple(
                int(part.strip())
                for part in candidate["StintLengths"].split("/")
            )

            result = simulate_strategy(
                stint_compounds=compounds,
                stint_lengths=lengths,
                pace_seconds=0.0,
                config=scenario_config,
            )

            scenario_rows.append({
                "CandidateId": candidate_id,
                "Stops": result["Stops"],
                "CompoundSequence": result["CompoundSequence"],
                "StintLengths": result["StintLengths"],
                "PitLossScenarioSeconds": float(pit_loss),
                "DegradationMultiplier": float(multiplier),
                "ScenarioRaceTimeSeconds": result[
                    "EstimatedRaceTimeSeconds"
                ],
            })

    details = pd.DataFrame(scenario_rows)

    # Compare candidates only within the same scenario.
    scenario_keys = [
        "PitLossScenarioSeconds",
        "DegradationMultiplier",
    ]

    details["ScenarioBestSeconds"] = details.groupby(
        scenario_keys
    )["ScenarioRaceTimeSeconds"].transform("min")

    details["RegretSeconds"] = (
        details["ScenarioRaceTimeSeconds"]
        - details["ScenarioBestSeconds"]
    )

    details["ScenarioRank"] = details.groupby(
        scenario_keys
    )["ScenarioRaceTimeSeconds"].rank(method="min")

    details["IsScenarioWinner"] = np.isclose(
        details["ScenarioRaceTimeSeconds"],
        details["ScenarioBestSeconds"],
        atol=1e-6,
    )

    # Share credit between candidates tied for the best result.
    winner_count = details.groupby(scenario_keys)[
        "IsScenarioWinner"
    ].transform("sum")

    details["FractionalWinCredit"] = (
        details["IsScenarioWinner"].astype(float) / winner_count
    )

    summary = details.groupby("CandidateId", as_index=False).agg(
        Stops=("Stops", "first"),
        CompoundSequence=("CompoundSequence", "first"),
        StintLengths=("StintLengths", "first"),
        ScenariosEvaluated=("ScenarioRaceTimeSeconds", "count"),
        MeanScenarioTimeSeconds=("ScenarioRaceTimeSeconds", "mean"),
        WorstCaseScenarioTimeSeconds=("ScenarioRaceTimeSeconds", "max"),
        MeanRegretSeconds=("RegretSeconds", "mean"),
        WorstCaseRegretSeconds=("RegretSeconds", "max"),
        MeanScenarioRank=("ScenarioRank", "mean"),
        ScenariosTiedForBest=("IsScenarioWinner", "sum"),
        FractionalScenarioWins=("FractionalWinCredit", "sum"),
    )

    summary["FractionalWinRate"] = (
        summary["FractionalScenarioWins"]
        / summary["ScenariosEvaluated"]
    )

    summary["ScenarioWinRate"] = (
        summary["ScenariosTiedForBest"]
        / summary["ScenariosEvaluated"]
    )

    return summary.sort_values(
        [
            "FractionalWinRate",
            "MeanRegretSeconds",
            "WorstCaseRegretSeconds",
        ],
        ascending=[False, True, True],
    ).reset_index(drop=True)

ROOT = Path(__file__).resolve().parent
enable_cache(ROOT / "data" / "cache")
st.set_page_config(page_title="PitWall AI", page_icon="🏎️", layout="wide")
st.title("PitWall AI")
st.caption("Formula 1 race intelligence — observed data, derived metrics and explicit uncertainty.")


with st.sidebar:
    st.header("Session")

    year = st.selectbox(
        "Season",
        list(range(2026, 2017, -1)),
        index=2,
    )

    @st.cache_data(ttl=3600)
    def cached_event_names(season: int) -> list[str]:
        return get_event_names(season)

    try:
        event_options = cached_event_names(int(year))

# Include configured upcoming races even when they have no session data yet.
        if int(year) == 2026:
            configured_events = [
                race["event"] for race in get_upcoming_races()
            ]
            event_options = list(dict.fromkeys(
                list(event_options) + configured_events
            ))

        if event_options:
            preferred_event = (
            "United States Grand Prix"
            if int(year) == 2026
            else "British Grand Prix"
        )

            default_event = next(
            (
                i for i, name in enumerate(event_options)
                if name == preferred_event
            ),
            0,
        )

            event = st.selectbox(
                "Grand Prix / event",
                event_options,
                index=default_event,
            )
        else:
            st.error("No Grand Prix events found for this season.")
            event = None

    except Exception as exc:
        st.error(f"Could not load the season calendar: {exc}")
        event = None

    session_name = st.selectbox(
        "Session",
        ["R", "Q", "FP3", "FP2", "FP1"],
    )

    load_clicked = st.button(
        "Load session",
        type="primary",
        use_container_width=True,
        disabled=event is None,
    )



if load_clicked:
    st.session_state.pop("pitwall_session", None)
    st.session_state.pop("pitwall_source", None)

    label = f"{year} · {event} · {session_name}"

    with st.spinner("Loading race data..."):
        fastf1_error = None

        if session_name == "R":
            try:
                candidate = load_session(
                    int(year), event.strip(), session_name
                )

                # FastF1 can finish loading without usable lap data.
                candidate_laps = candidate.laps.copy()

                if candidate_laps.empty:
                    raise ValueError("FastF1 returned empty lap data.")

                st.session_state["pitwall_session"] = candidate
                st.session_state["pitwall_source"] = "FastF1"
                st.session_state["pitwall_label"] = label

            except Exception as exc:
                fastf1_error = exc

        else:
            fastf1_error = (
                "Jolpica fallback currently supports race sessions only."
            )

        if "pitwall_session" not in st.session_state:
            try:
                round_number = find_round_number(
                    int(year), event.strip()
                )
                fallback_laps = load_laps(
                    int(year), round_number
                )

                if fallback_laps.empty:
                    raise ValueError(
                        "Jolpica returned no lap records."
                    )

                st.session_state["pitwall_session"] = fallback_laps
                st.session_state["pitwall_source"] = "Jolpica API"
                st.session_state["pitwall_label"] = label

                st.warning(
                    "FastF1 lap data was unavailable. "
                    "Using Jolpica historical data instead. "
                    "Tyre, weather and telemetry data may be unavailable."
                )

            except Exception as fallback_error:
                st.error(
                    f"FastF1 failed: {fastf1_error}\n\n"
                    f"Jolpica fallback failed: {fallback_error}"
                )



session = st.session_state.get("pitwall_session")

if session is None:
    st.title("Pre-Race Strategy Planner")
    st.caption(
        "Explore strategy scenarios before a race session is available. "
        "Results depend on explicit illustrative assumptions."
    )

    try:
        race = get_race_by_event(event)
    except (ValueError, NameError):
        race = None

    if race is None:
        st.info(
            "No session is loaded. Select a configured upcoming race "
            "and use the pre-race planner."
        )
        st.stop()

    st.subheader(f"{race['event']} — {race['circuit']}")
    st.write(
        f"Scheduled date: {race['race_date']} · "
        f"Race distance: {race['total_laps']} laps"
    )

    if race.get("pit_loss_is_estimate", True):
        st.warning(
            "Pit-stop loss is an estimate, not a verified circuit measurement."
        )

    left, right = st.columns(2)

    with left:
        total_laps = st.number_input(
            "Race distance (laps)",
            min_value=10,
            max_value=100,
            value=int(race["total_laps"]),
            step=1,
            key="pr_total_laps",
        )
        pit_loss = st.slider(
            "Pit-stop time loss (seconds)",
            min_value=10.0,
            max_value=35.0,
            value=float(race["pit_loss_seconds"]),
            step=0.5,
            key="pr_pit_loss",
        )

    with right:
        max_stops = st.selectbox(
            "Maximum pit stops",
            [1, 2, 3],
            index=2,
            key="pr_max_stops",
        )
        min_stint = st.slider(
            "Minimum stint length (laps)",
            min_value=3,
            max_value=10,
            value=5,
            key="pr_min_stint",
        )

    st.markdown("#### Illustrative compound assumptions")
    st.caption(
        "These values are model inputs, not official Pirelli measurements. "
        "They should be calibrated against suitable historical tyre data."
    )

    compounds = ["SOFT", "MEDIUM", "HARD"]
    default_pace = {"SOFT": 0.0, "MEDIUM": 0.35, "HARD": 0.65}
    default_deg = {"SOFT": 0.075, "MEDIUM": 0.045, "HARD": 0.030}
    pace_inputs = {}
    deg_inputs = {}

    cols = st.columns(3)
    for i, compound in enumerate(compounds):
        with cols[i]:
            st.markdown(f"**{compound}**")
            pace_inputs[compound] = st.number_input(
                "Pace delta (s/lap)",
                min_value=-2.0,
                max_value=3.0,
                value=default_pace[compound],
                step=0.05,
                key=f"pr_pace_{compound}",
            )
            deg_inputs[compound] = st.number_input(
                "Degradation (s/tyre-age lap)",
                min_value=0.0,
                max_value=0.5,
                value=default_deg[compound],
                step=0.005,
                format="%.3f",
                key=f"pr_deg_{compound}",
            )

    if st.button("Generate strategy candidates", type="primary"):
        try:
            config = StrategyConfig(
                total_laps=int(total_laps),
                pit_loss_seconds=float(pit_loss),
                min_stint_laps=int(min_stint),
                max_stops=int(max_stops),
                compound_pace_delta=tuple(pace_inputs.items()),
                compound_degradation=tuple(deg_inputs.items()),
            )

            # Neutral baseline for comparing strategy assumptions.
            # This is not a measured lap time or a race-time forecast.
            candidates = generate_candidate_strategies(
                driver_pace_seconds=0.0,
                config=config,
                pit_window_step=2,
                retain_top_n=10,
            )

            if candidates.empty:
                st.warning(
                    "No feasible candidates were generated. "
                    "Try reducing the minimum stint length or stop limit."
                )
            else:
                st.session_state["pre_race_candidates"] = candidates
                st.session_state["pre_race_event"] = race["event"]
                st.session_state["pre_race_config"] = {
                    "laps": int(total_laps),
                    "pit_loss": float(pit_loss),
                    "min_stint": int(min_stint),
                    "max_stops": int(max_stops),
                }

        except Exception as exc:
            st.error(f"Could not generate strategies: {exc}")

    candidates = st.session_state.get(
        "pre_race_candidates", pd.DataFrame()
    )

    if (
        not candidates.empty
        and st.session_state.get("pre_race_event") == race["event"]
    ):
        st.markdown("#### Best candidate for each stop count")

        best_by_stops = (
            candidates.sort_values("EstimatedRaceTimeSeconds")
            .groupby("Stops", as_index=False)
            .first()
            .sort_values("Stops")
        )

        st.dataframe(
            best_by_stops[
                [
                    "Stops",
                    "CompoundSequence",
                    "StintLengths",
                    "EstimatedRaceTimeSeconds",
                    "EstimatedPitLossSeconds",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Top candidate strategies")

        st.dataframe(
            candidates.head(10),
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            candidates.head(10),
            x="EstimatedRaceTimeSeconds",
            y="CompoundSequence",
            color="Stops",
            orientation="h",
            hover_data=["StintLengths", "EstimatedPitLossSeconds"],
            title="Candidate ranking under selected assumptions",
        )
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

        
        st.markdown("### Strategy robustness analysis")
        st.caption(
            "Compare the shortlisted strategies across 9 scenarios: "
            "pit loss of 18, 23 and 28 seconds, combined with "
            "degradation multipliers of 0.75, 1.00 and 1.25."
        )

        if st.button(
            "Run robustness analysis",
            key="active_pre_race_robustness",
        ):
            try:
                saved_config = st.session_state.get(
                    "pre_race_config", {}
                )

                robustness_config = StrategyConfig(
                    total_laps=int(saved_config.get("laps", total_laps)),
                    pit_loss_seconds=float(
                        saved_config.get("pit_loss", pit_loss)
                    ),
                    min_stint_laps=int(
                        saved_config.get("min_stint", min_stint)
                    ),
                    max_stops=int(
                        saved_config.get("max_stops", max_stops)
                    ),
                    compound_pace_delta=tuple(pace_inputs.items()),
                    compound_degradation=tuple(deg_inputs.items()),
                )

                robust_results = evaluate_candidate_robustness(
                    candidates=candidates,
                    base_config=robustness_config,
                )

                st.session_state["active_robustness_results"] = (
                    robust_results
                )
                st.session_state["active_robustness_event"] = (
                    race["event"]
                )

            except Exception as exc:
                st.error(f"Robustness analysis failed: {exc}")

        robust_results = st.session_state.get(
            "active_robustness_results", pd.DataFrame()
        )

        if (
            not robust_results.empty
            and st.session_state.get("active_robustness_event")
            == race["event"]
        ):
            st.markdown("#### Robustness results")
            st.dataframe(
                robust_results,
                use_container_width=True,
                hide_index=True,
            )
            
            
            # Select the most robust strategy using the existing ranking.
            recommended = robust_results.iloc[0]

            st.markdown("#### Recommended robust strategy")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Fractional win rate",
                    f"{recommended['FractionalWinRate']:.1%}",
                )

            with col2:
                st.metric(
                    "Mean regret",
                    f"{recommended['MeanRegretSeconds']:.2f} s",
                )

            with col3:
                st.metric(
                    "Worst-case regret",
                    f"{recommended['WorstCaseRegretSeconds']:.2f} s",
                )

            st.write(
                f"**Tyre sequence:** {recommended['CompoundSequence']}"
            )
            st.write(
                f"**Stint lengths:** {recommended['StintLengths']}"
            )
            st.write(
                f"**Number of stops:** {int(recommended['Stops'])}"
            )

            # Prepare one interactive robustness chart.
            plot_data = robust_results.copy()
            plot_data["Strategy"] = (
                plot_data["CompoundSequence"]
                + " | "
                + plot_data["StintLengths"].astype(str)
            )
            plot_data["Recommendation"] = "Other candidates"
            plot_data.loc[plot_data.index[0], "Recommendation"] = (
                "Recommended strategy"
            )

            fig = px.scatter(
                plot_data,
                x="MeanRegretSeconds",
                y="FractionalWinRate",
                color="Recommendation",
                symbol="Recommendation",
                size="ScenariosTiedForBest",
                hover_name="Strategy",
                hover_data={
                    "Stops": True,
                    "FractionalWinRate": ":.1%",
                    "MeanRegretSeconds": ":.2f",
                    "WorstCaseRegretSeconds": ":.2f",
                    "MeanScenarioRank": ":.2f",
                    "ScenariosEvaluated": True,
                    "Recommendation": True,
                },
                title="Strategy robustness: win rate vs. regret",
                labels={
                    "MeanRegretSeconds": "Mean regret (seconds)",
                    "FractionalWinRate": "Fractional win rate",
                    "Stops": "Number of pit stops",
                    "ScenariosTiedForBest": "Scenarios tied for best",
                },
            )

            # Zoom to the observed data while keeping a little margin.
            x_min = float(plot_data["MeanRegretSeconds"].min())
            x_max = float(plot_data["MeanRegretSeconds"].max())
            y_min = float(plot_data["FractionalWinRate"].min())
            y_max = float(plot_data["FractionalWinRate"].max())

            x_margin = max((x_max - x_min) * 0.12, 0.25)
            y_margin = max((y_max - y_min) * 0.15, 0.02)

            fig.update_xaxes(
                range=[
                    max(0, x_min - x_margin),
                    x_max + x_margin,
                ],
                title="Mean regret (seconds — lower is better)",
                zeroline=True,
                rangeslider_visible=False,
            )
            fig.update_yaxes(
                range=[
                    max(0.0, y_min - y_margin),
                    min(1.0, y_max + y_margin),
                ],
                title="Fractional win rate (higher is better)",
            )
            fig.update_layout(
                height=520,
                margin=dict(l=30, r=180, t=65, b=40),
                legend=dict(
                    title="Strategy details",
                    orientation="v",
                    x=1.02,
                    y=1,
                    xanchor="left",
                    yanchor="top",
                    font=dict(size=11),
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="active_pre_race_robustness_chart",
            )
            
            st.markdown("#### Compare leading strategies")

            comparison_columns = [
                "Stops",
                "CompoundSequence",
                "StintLengths",
                "FractionalWinRate",
                "MeanRegretSeconds",
                "WorstCaseRegretSeconds",
                "MeanScenarioRank",
            ]

            comparison = robust_results[comparison_columns].copy()

            comparison = comparison.rename(
                columns={
                    "Stops": "Pit stops",
                    "CompoundSequence": "Tyre sequence",
                    "StintLengths": "Stint lengths",
                    "FractionalWinRate": "Fractional win rate",
                    "MeanRegretSeconds": "Mean regret (s)",
                    "WorstCaseRegretSeconds": "Worst-case regret (s)",
                    "MeanScenarioRank": "Mean scenario rank",
                }
            )

            comparison["Fractional win rate"] = (
                comparison["Fractional win rate"].map(
                    lambda value: f"{value:.1%}"
                )
            )

            comparison["Mean regret (s)"] = (
                comparison["Mean regret (s)"].round(2)
            )

            comparison["Worst-case regret (s)"] = (
                comparison["Worst-case regret (s)"].round(2)
            )

            comparison["Mean scenario rank"] = (
                comparison["Mean scenario rank"].round(2)
            )

            st.dataframe(
                comparison.head(5),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Prefer strategies toward the upper-left: higher fractional "
                "win rate and lower mean regret. The recommendation applies "
                "only to the tested shortlist and model assumptions; it is "
                "not a guarantee of race outcome."
            )

        st.warning(
            "These are assumption-based scenario scores, not predicted "
            "race finishing times. A zero-second baseline pace is used "
            "to compare strategies; fuel, traffic, track position, weather, "
            "safety cars, tyre allocation rules and driver-specific pace "
            "are not modelled in this pre-race mode. Pit windows are sampled "
            "in two-lap steps."
        )


    st.stop()



try:
    state = build_race_state(session)

    if isinstance(session, pd.DataFrame):
        try:
            race_date = "2024-07-07"

            if "Year" in session.columns and "Round" in session.columns:
                year_value = int(session["Year"].iloc[0])
                round_value = int(session["Round"].iloc[0])

                from src.pitwall.data.jolpica_loader import BASE_URL
                import requests

                url = f"{BASE_URL}/{year_value}/{round_value}.json"
                response = requests.get(url, timeout=20)
                response.raise_for_status()

                races = response.json()["MRData"]["RaceTable"]["Races"]
                if races:
                    race_date = races[0]["date"]

            weather = load_historical_weather(race_date)

            state["weather_samples"] = weather

            summary_rows = []
            for column in [
                "AirTemp", "Humidity", "Rainfall",
                "WindSpeed", "WindGusts", "CloudCover",
            ]:
                if column in weather.columns:
                    values = pd.to_numeric(
                        weather[column], errors="coerce"
                    ).dropna()

                    if not values.empty:
                        summary_rows.append({
                            "Metric": column,
                            "Min": round(float(values.min()), 2),
                            "Mean": round(float(values.mean()), 2),
                            "Max": round(float(values.max()), 2),
                        })

            state["weather_summary"] = pd.DataFrame(summary_rows)
            state["availability"]["Weather samples"] = not weather.empty

            state["caveats"].append(
                "Weather is historical reanalysis near the circuit, "
                "not verified trackside weather-station data."
            )

        except Exception as weather_error:
            state["weather_samples"] = pd.DataFrame()
            state["weather_summary"] = pd.DataFrame()
            state["availability"]["Weather samples"] = False

            st.warning(
                f"Historical weather could not be loaded: {weather_error}"
            )

except Exception as exc:
    st.error(f"Could not build race state: {exc}")
    st.stop()


st.subheader(st.session_state.get("pitwall_label", "Loaded session"))
st.caption(
    f"Data source: {st.session_state.get('pitwall_source', 'Unknown')}"
)
a, b, c, d = st.columns(4)
a.metric("Lap records", f"{state['lap_count']:,}")
b.metric("Valid timed laps", f"{state['valid_lap_count']:,}")
c.metric("Drivers", f"{state['driver_count']:,}")
d.metric("Weather samples", f"{len(state['weather_samples']):,}")

overview, pace_tab, tyres_tab, weather_tab, results_tab, strategy_tab = st.tabs(
    [
        "Overview",
        "Driver & car pace",
        "Tyres & stints",
        "Weather",
        "Results & data quality",
        "Strategy simulator",
    ]
)
with overview:
    st.markdown("### Unified race-state snapshot")
    st.json(state["event"])
    st.dataframe(pd.DataFrame([{"Data source": k, "Available": v} for k, v in state["availability"].items()]),
                 use_container_width=True, hide_index=True)
    st.write("This milestone summarizes available historical session data; it does not yet make strategy recommendations.")

with pace_tab:
    st.markdown("### Observed driver pace")
    pace = state["driver_pace"]
    if pace.empty:
        st.warning("No valid lap times available.")
    else:
        st.dataframe(pace, use_container_width=True, hide_index=True)
        metric = st.selectbox("Compare metric", ["MedianLapSeconds", "BestLapSeconds", "StdDevSeconds", "MedianGapToFastestSeconds"])
        fig = px.bar(pace.sort_values(metric), x="Driver", y=metric, title=f"Driver comparison: {metric}")
        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("### Lap-by-lap pace")

    loaded = st.session_state.get("pitwall_session")

    if isinstance(loaded, pd.DataFrame):
        trend_data = loaded.copy()
        trend_data["LapTimeSeconds"] = pd.to_numeric(
            trend_data["LapTime"], errors="coerce"
        )
    else:
        trend_data = state.get("lap_data", pd.DataFrame()).copy()

    if not trend_data.empty and {
        "Driver", "LapNumber", "LapTimeSeconds"
    }.issubset(trend_data.columns):
        trend_data = trend_data[
            trend_data["LapTimeSeconds"].between(
                40, 200, inclusive="both"
            )
        ]

        available_drivers = sorted(
            trend_data["Driver"].dropna().unique().tolist()
        )

        selected_drivers = st.multiselect(
            "Drivers to compare",
            available_drivers,
            default=available_drivers[:3],
            key="pace_trend_drivers",
        )

        plot_data = trend_data[
            trend_data["Driver"].isin(selected_drivers)
        ]

        if not plot_data.empty:
            fig = px.line(
                plot_data,
                x="LapNumber",
                y="LapTimeSeconds",
                color="Driver",
                markers=False,
                title="Lap time by lap",
                labels={
                    "LapNumber": "Lap",
                    "LapTimeSeconds": "Lap time (seconds)",
                },
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one driver to plot lap times.")
    else:
        st.info("Lap-by-lap timing data is unavailable.")

    st.caption("This is not a pure driver ranking: car, traffic, fuel, setup, tyre age and track conditions confound pace.")



with tyres_tab:
    st.markdown("### Tyre and stint analysis")

    loaded = st.session_state.get("pitwall_session")

    if isinstance(loaded, pd.DataFrame):
        year_value = int(loaded["Year"].iloc[0])
        round_value = int(loaded["Round"].iloc[0])

        try:
            stops = load_pit_stops(year_value, round_value)

            if stops.empty:
                st.info("No pit-stop records were returned.")
            else:
                counts = (
                    stops.groupby("Driver")
                    .size()
                    .rename("PitStops")
                    .reset_index()
                    .sort_values("PitStops", ascending=False)
                )

                left, right = st.columns(2)
                left.metric("Recorded pit stops", len(stops))
                right.metric(
                    "Drivers with pit stops",
                    stops["Driver"].nunique(),
                )

                st.markdown("#### Pit stops by driver")
                st.dataframe(
                    counts,
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("#### Pit-stop timeline")
                fig = px.scatter(
                    stops,
                    x="LapNumber",
                    y="Driver",
                    color="Driver",
                    hover_data=["Stop", "PitDuration"],
                    title="Recorded pit stops by race lap",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Estimated stint intervals")
                intervals = []

                for driver, group in stops.groupby("Driver"):
                    stop_laps = sorted(
                        group["LapNumber"].dropna().astype(int).tolist()
                    )

                    # A stop on lap N normally means the next stint
                    # begins on or around lap N; exact timing varies.
                    starts = [1] + stop_laps

                    for index, start_lap in enumerate(starts):
                        end_lap = (
                            starts[index + 1] - 1
                            if index + 1 < len(starts)
                            else None
                        )

                        intervals.append({
                            "Driver": driver,
                            "Stint": index + 1,
                            "StartLap": start_lap,
                            "EndLap": end_lap,
                            "Compound": "Unavailable",
                            "TyreAge": "Unavailable",
                        })

                interval_df = pd.DataFrame(intervals)

                if not interval_df.empty:
                    st.dataframe(
                        interval_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                st.warning(
                    "Jolpica provides pit-stop records, not tyre compounds "
                    "or tyre age. Stint boundaries are approximate."
                )

        except Exception as exc:
            st.error(f"Could not load pit-stop data: {exc}")

    else:
        # FastF1 provides compound and tyre-life data when available.
        stints = state["tyre_stints"]

        if stints.empty:
            st.warning("Tyre/stint data is unavailable for this session.")
        else:
            st.markdown("#### Stint summary")
            st.dataframe(
                stints,
                use_container_width=True,
                hide_index=True,
            )

        timeline = loaded.laps.copy()

        required = {"Driver", "Stint", "LapNumber", "Compound"}

        if required.issubset(timeline.columns):
            timeline = timeline.dropna(
                subset=["Driver", "Stint", "LapNumber", "Compound"]
            ).copy()

            timeline["LapNumber"] = pd.to_numeric(
                timeline["LapNumber"], errors="coerce"
            )
            timeline["Stint"] = pd.to_numeric(
                timeline["Stint"], errors="coerce"
            )
            timeline = timeline.dropna(
                subset=["LapNumber", "Stint"]
            )

            timeline["Stint"] = timeline["Stint"].astype(int)

            timeline = (
                timeline.groupby(
                    ["Driver", "Stint", "Compound"],
                    as_index=False,
                )
                .agg(
                    StartLap=("LapNumber", "min"),
                    EndLap=("LapNumber", "max"),
                )
            )

            drivers = sorted(timeline["Driver"].unique())

            selected = st.multiselect(
                "Drivers to show",
                drivers,
                default=drivers[:5],
                key="stint_timeline_drivers",
            )

            plot_data = timeline[
                timeline["Driver"].isin(selected)
            ]

            st.markdown("#### Tyre stints across the race")

            if not plot_data.empty:
                
                fig = px.bar(
                    plot_data,
                    x=plot_data["EndLap"] - plot_data["StartLap"] + 1,
                    base="StartLap",
                    y="Driver",
                    color="Compound",
                    orientation="h",
                    hover_data=["Stint", "StartLap", "EndLap"],
                    title="Tyre stints across the race",
                    labels={
                        "x": "Race laps",
                        "Driver": "Driver",
                        "Compound": "Tyre compound",
                    },
                )

                fig.update_layout(
                    xaxis_title="Race lap",
                    yaxis_title="Driver",
                    legend_title="Compound",
                )

                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.info("Select at least one driver.")
        else:
            st.info("Compound and stint fields are unavailable.")
    
    st.markdown("### Tyre degradation analysis")

    lap_data = state.get("lap_data", pd.DataFrame()).copy()

    required_columns = {
        "Driver", "Compound", "TyreLife", "LapTimeSeconds"
    }

    if not required_columns.issubset(lap_data.columns):
        st.info(
            "Tyre degradation analysis requires FastF1 tyre-age data. "
            "It is unavailable for the current Jolpica-only dataset."
        )
    else:
        lap_data["TyreLife"] = pd.to_numeric(
            lap_data["TyreLife"], errors="coerce"
        )
        lap_data["LapTimeSeconds"] = pd.to_numeric(
            lap_data["LapTimeSeconds"], errors="coerce"
        )

        # Remove missing values and implausible lap times.
        degradation = lap_data.dropna(
            subset=[
                "Driver", "Compound", "TyreLife", "LapTimeSeconds"
            ]
        ).copy()

        degradation = degradation[
            degradation["TyreLife"].between(1, 80)
            & degradation["LapTimeSeconds"].between(
                40, 200, inclusive="both"
            )
        ]

        # Exclude pit-in and pit-out laps where the data provides them.
        if "PitInTime" in degradation.columns:
            degradation = degradation[
                degradation["PitInTime"].isna()
            ]

        if "PitOutTime" in degradation.columns:
            degradation = degradation[
                degradation["PitOutTime"].isna()
            ]

        if degradation.empty:
            st.warning("No usable tyre-age and lap-time records were found.")
        else:
            compounds = sorted(
                degradation["Compound"].dropna().unique().tolist()
            )

            selected_compounds = st.multiselect(
                "Compounds to compare",
                compounds,
                default=compounds,
                key="degradation_compounds",
            )

            available_drivers = sorted(
                degradation["Driver"].dropna().unique().tolist()
            )

            selected_drivers = st.multiselect(
                "Drivers for degradation analysis",
                available_drivers,
                default=available_drivers[:3],
                key="degradation_drivers",
            )

            filtered = degradation[
                degradation["Compound"].isin(selected_compounds)
                & degradation["Driver"].isin(selected_drivers)
            ]

            if not filtered.empty:
                # Median per driver, compound and tyre age.
                trend = (
                    filtered.groupby(
                        ["Driver", "Compound", "TyreLife"],
                        as_index=False,
                    )
                    .agg(
                        MedianLapSeconds=("LapTimeSeconds", "median"),
                        LapCount=("LapTimeSeconds", "size"),
                    )
                )

                fig = px.line(
                    trend,
                    x="TyreLife",
                    y="MedianLapSeconds",
                    color="Compound",
                    line_dash="Driver",
                    markers=True,
                    hover_data=["LapCount"],
                    title="Lap time versus tyre age",
                    labels={
                        "TyreLife": "Tyre age (laps)",
                        "MedianLapSeconds": "Median lap time (seconds)",
                        "Compound": "Tyre compound",
                        "Driver": "Driver",
                    },
                )

                st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "Each point is a median for one driver, compound "
                    "and tyre age. Higher lap times mean slower laps; "
                    "they do not by themselves prove tyre degradation."
                )
            else:
                st.info(
                    "Select at least one compound and one driver."
                )


    st.caption(
        "Stint summaries are descriptive. Tyre degradation modelling "
        "and pit-window prediction are not implemented yet."
    )



with weather_tab:
    st.markdown("### Recorded weather samples")
    weather = state["weather_samples"]
    if state["weather_summary"].empty: st.warning("No usable weather measurements returned.")
    else: st.dataframe(state["weather_summary"], use_container_width=True, hide_index=True)
    if not weather.empty:
        choices = [c for c in ["AirTemp", "TrackTemp", "Humidity", "WindSpeed", "Rainfall"] if c in weather]
        if choices and "Time" in weather:
            metric = st.selectbox("Weather variable", choices)
            st.plotly_chart(px.line(weather, x="Time", y=metric, title=f"{metric} during session"), use_container_width=True)
    st.caption("These are recorded samples, not live conditions or a forecast.")

with results_tab:
    st.markdown("### Session results")
    if state["results"].empty: st.warning("No result table available.")
    else: st.dataframe(state["results"], use_container_width=True, hide_index=True)
    st.markdown("### Data quality and limitations")
    for note in state["caveats"]: st.write(f"- {note}")
    
    with st.expander("Raw data columns"):
        loaded = st.session_state.get("pitwall_session")

        if isinstance(loaded, pd.DataFrame):
            st.write("Jolpica lap columns")
            st.code(", ".join(map(str, loaded.columns)))
        else:
            try:
                st.write("Lap columns")
                st.code(", ".join(map(str, loaded.laps.columns)))
            except Exception:
                st.info("Lap columns are unavailable.")

            try:
                st.write("Weather columns")
                st.code(", ".join(map(str, loaded.weather_data.columns)))
            except Exception:
                st.info("Weather columns are unavailable.")

            try:
                st.write("Result columns")
                st.code(", ".join(map(str, loaded.results.columns)))
            except Exception:
                st.info("Result columns are unavailable.")




with strategy_tab:
    st.markdown("### Multi-stop strategy optimizer")
    st.caption(
        "Compare pit-stop strategies using historical tyre data when "
        "available, or explicit assumptions when it is not."
    )

    loaded = st.session_state.get("pitwall_session")
    laps = state.get("lap_data", pd.DataFrame()).copy()

    # Resolve configured race metadata, if available.
    try:
        race = get_race_by_event(event)
    except (ValueError, NameError):
        race = None

    required = {
        "Driver", "LapNumber", "Compound",
        "TyreLife", "LapTimeSeconds",
    }

    has_tyre_data = (
        not isinstance(loaded, pd.DataFrame)
        and not laps.empty
        and required.issubset(laps.columns)
        and laps["Compound"].notna().any()
        and pd.to_numeric(
            laps["TyreLife"], errors="coerce"
        ).notna().any()
    )
    st.write("DEBUG — loaded type:", type(loaded).__name__)
    st.write("DEBUG — laps shape:", laps.shape)
    st.write("DEBUG — columns:", list(laps.columns))
    st.write("DEBUG — has_tyre_data:", has_tyre_data)

    if has_tyre_data:
        # ---------------------------------------------------------
        # HISTORICAL MODE: use observed FastF1 tyre data.
        # ---------------------------------------------------------
        st.success(
            "Historical tyre data is available. "
            "The optimizer will fit compound models to observed laps."
        )

        max_lap_value = pd.to_numeric(
            laps["LapNumber"], errors="coerce"
        ).max()

        default_laps = (
            int(race["total_laps"])
            if race
            else int(max_lap_value)
        )

        driver_options = sorted(
            laps["Driver"].dropna().unique().tolist()
        )

        left, right = st.columns(2)

        with left:
            total_laps = st.number_input(
                "Race distance (laps)",
                min_value=10,
                max_value=100,
                value=default_laps,
                step=1,
                key="historical_total_laps",
            )

            default_pit_loss = (
                float(race["pit_loss_seconds"])
                if race
                else 22.0
            )

            pit_loss = st.slider(
                "Pit-stop time loss (seconds)",
                min_value=10.0,
                max_value=35.0,
                value=default_pit_loss,
                step=0.5,
                key="historical_pit_loss",
            )

        with right:
            max_stops = st.selectbox(
                "Maximum pit stops",
                [1, 2, 3],
                index=2,
                key="historical_max_stops",
            )

            min_stint = st.slider(
                "Minimum stint length (laps)",
                min_value=3,
                max_value=10,
                value=5,
                key="historical_min_stint",
            )

        selected_driver = st.selectbox(
            "Reference driver",
            driver_options,
            key="optimizer_reference_driver",
        )

        if st.button(
            "Optimize from historical tyre data",
            type="primary",
            key="historical_optimize",
        ):
            from src.pitwall.analytics.strategy import (
                estimate_compound_models,
            )

            driver_data = laps[
                laps["Driver"] == selected_driver
            ].copy()

            driver_models = estimate_compound_models(driver_data)

            if len(driver_models) >= 2:
                model_data = driver_data
                model_scope = f"Reference driver: {selected_driver}"
            else:
                model_data = laps
                model_scope = "Pooled data from the session"

            strategies, note = optimize_strategies(
                laps=model_data,
                total_laps=int(total_laps),
                pit_loss_seconds=float(pit_loss),
                max_stops=int(max_stops),
                min_stint_laps=int(min_stint),
            )
            models = estimate_compound_models(model_data)

            st.markdown("#### Tyre calibration diagnostics")
            st.dataframe(
                models,
                use_container_width=True,
                hide_index=True,
            )

            st.session_state["optimized_strategies"] = strategies
            st.session_state["strategy_optimizer_note"] = note
            st.session_state["strategy_model_scope"] = model_scope

        strategies = st.session_state.get(
            "optimized_strategies", pd.DataFrame()
        )

        if not strategies.empty:
            st.markdown("#### Best strategy by stop count")

            best_by_stops = (
                strategies.sort_values("EstimatedRaceTimeSeconds")
                .groupby("Stops", as_index=False)
                .first()
                .sort_values("Stops")
            )

            st.dataframe(
                best_by_stops[
                    [
                        "Stops",
                        "CompoundSequence",
                        "StintLengths",
                        "EstimatedRaceTimeSeconds",
                        "PitLossSeconds",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Top 10 historical-model candidates")

            top_strategies = strategies.head(10)

            st.dataframe(
                top_strategies[
                    [
                        "Stops",
                        "CompoundSequence",
                        "StintLengths",
                        "EstimatedRaceTimeSeconds",
                        "PitLossSeconds",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            fig = px.bar(
                top_strategies,
                x="EstimatedRaceTimeSeconds",
                y="CompoundSequence",
                color="Stops",
                orientation="h",
                title="Historical-model strategy comparison",
                hover_data=["StintLengths", "PitLossSeconds"],
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Model scope: "
                + st.session_state.get(
                    "strategy_model_scope", "Unknown"
                )
            )
            st.info(
                st.session_state.get("strategy_optimizer_note", "")
            )
            
            st.markdown("#### Strategy robustness analysis")

            robust_strategies = analyze_strategy_robustness(strategies)

            if not robust_strategies.empty:
                st.caption(
                    "Strategies are compared across 9 combinations "
                    "of pit-stop loss and degradation assumptions. "
                    "This is a sensitivity analysis, not a forecast."
                )

                st.dataframe(
                    robust_strategies[
                        [
                            "Stops",
                            "CompoundSequence",
                            "StintLengths",
                            "MeanScenarioRank",
                            "WorstCaseRank",
                            "WorstCaseTimeSeconds",
                            "ScenarioWinRate",
                        ]
                    ].head(10),
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("##### Most robust candidate")
                best = robust_strategies.iloc[[0]][
                    [
                        "Stops",
                        "CompoundSequence",
                        "StintLengths",
                        "MeanScenarioRank",
                        "WorstCaseRank",
                        "ScenarioWinRate",
                    ]
                ]

                st.dataframe(
                    best,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.warning(
                    "Robustness analysis is unavailable for these "
                    "strategies. Check that the strategy model returns "
                    "baseline and degradation time components."
                )

        else:
            st.info(
                "Click 'Optimize from historical tyre data' "
                "to calculate candidate strategies."
            )


    else:
        # ---------------------------------------------------------
        # FALLBACK MODE: historical tyre data is unavailable.
        # Works for completed races and upcoming races.
        # ---------------------------------------------------------
        st.info(
            "Tyre compound or tyre-age data is unavailable for this "
            "session. You can still compare strategies using "
            "illustrative assumptions."
        )

        if race:
            st.markdown(
                f"**{race['event']} — {race['circuit']}**"
            )
            st.caption(
                f"Scheduled date: {race['race_date']} · "
                f"Configured distance: {race['total_laps']} laps"
            )
            default_laps = int(race["total_laps"])
            default_pit_loss = float(race["pit_loss_seconds"])
        else:
            observed_laps = pd.to_numeric(
                laps["LapNumber"], errors="coerce"
            ).max() if (
                not laps.empty and "LapNumber" in laps.columns
            ) else 0

            default_laps = (
                int(observed_laps)
                if pd.notna(observed_laps) and observed_laps >= 10
                else 56
            )
            default_pit_loss = 22.0

        if race and race.get("pit_loss_is_estimate", True):
            st.warning(
                "Pit-stop loss is an estimate, not a verified "
                "measurement for this circuit."
            )

        left, right = st.columns(2)

        with left:
            total_laps = st.number_input(
                "Race distance (laps)",
                min_value=10,
                max_value=100,
                value=default_laps,
                step=1,
                key="fallback_total_laps",
            )

            pit_loss = st.slider(
                "Pit-stop time loss (seconds)",
                min_value=10.0,
                max_value=35.0,
                value=default_pit_loss,
                step=0.5,
                key="fallback_pit_loss",
            )

        with right:
            max_stops = st.selectbox(
                "Maximum pit stops",
                [1, 2, 3],
                index=2,
                key="fallback_max_stops",
            )

            min_stint = st.slider(
                "Minimum stint length (laps)",
                min_value=3,
                max_value=10,
                value=5,
                key="fallback_min_stint",
            )

        compounds = ["SOFT", "MEDIUM", "HARD"]

        default_pace = {
            "SOFT": 0.0,
            "MEDIUM": 0.35,
            "HARD": 0.65,
        }

        default_degradation = {
            "SOFT": 0.075,
            "MEDIUM": 0.045,
            "HARD": 0.030,
        }

        st.markdown("#### Illustrative compound assumptions")
        st.caption(
            "These are editable model inputs, not official tyre data. "
            "They have not been validated as a forecast for this race."
        )

        pace_inputs = {}
        degradation_inputs = {}
        columns = st.columns(3)

        for i, compound in enumerate(compounds):
            with columns[i]:
                st.markdown(f"**{compound}**")

                pace_inputs[compound] = st.number_input(
                    "Pace delta (s/lap)",
                    min_value=-2.0,
                    max_value=3.0,
                    value=default_pace[compound],
                    step=0.05,
                    key=f"fallback_pace_{compound}",
                )

                degradation_inputs[compound] = st.number_input(
                    "Degradation (s/tyre-age lap)",
                    min_value=0.0,
                    max_value=0.5,
                    value=default_degradation[compound],
                    step=0.005,
                    format="%.3f",
                    key=f"fallback_degradation_{compound}",
                )

        if st.button(
            "Generate strategy candidates",
            type="primary",
            key="fallback_generate",
        ):
            try:
                config = StrategyConfig(
                    total_laps=int(total_laps),
                    pit_loss_seconds=float(pit_loss),
                    min_stint_laps=int(min_stint),
                    max_stops=int(max_stops),
                    compound_pace_delta=tuple(
                        pace_inputs.items()
                    ),
                    compound_degradation=tuple(
                        degradation_inputs.items()
                    ),
                )

                candidates = generate_candidate_strategies(
                    driver_pace_seconds=0.0,
                    config=config,
                    pit_window_step=2,
                    retain_top_n=10,
                )

                st.session_state["fallback_candidates"] = candidates
                st.session_state["fallback_event"] = event

                # Keep settings used to generate the displayed results.
                st.session_state["fallback_settings"] = {
                    "total_laps": int(total_laps),
                    "pit_loss": float(pit_loss),
                    "max_stops": int(max_stops),
                    "min_stint": int(min_stint),
                }

            except Exception as exc:
                st.error(f"Could not generate strategies: {exc}")

        candidates = st.session_state.get(
            "fallback_candidates", pd.DataFrame()
        )
        if (
            not candidates.empty
            and st.session_state.get("fallback_event") == event
        ):
            st.markdown("#### Best candidate by stop count")

            best_by_stops = (
                candidates.sort_values("EstimatedRaceTimeSeconds")
                .groupby("Stops", as_index=False)
                .first()
                .sort_values("Stops")
            )

            display_best = best_by_stops.rename(
                columns={
                    "EstimatedRaceTimeSeconds": "ScenarioScoreSeconds",
                    "EstimatedPitLossSeconds": "AssumedPitLossSeconds",
                }
            )

            st.dataframe(
                display_best[
                    [
                        "Stops",
                        "CompoundSequence",
                        "StintLengths",
                        "ScenarioScoreSeconds",
                        "AssumedPitLossSeconds",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Top candidate strategies")

            top_candidates = candidates.head(10).rename(
                columns={
                    "EstimatedRaceTimeSeconds": "ScenarioScoreSeconds",
                    "EstimatedPitLossSeconds": "AssumedPitLossSeconds",
                }
            )

            st.dataframe(
                top_candidates,
                use_container_width=True,
                hide_index=True,
            )

            fig = px.bar(
                candidates.head(10),
                x="EstimatedRaceTimeSeconds",
                y="CompoundSequence",
                color="Stops",
                orientation="h",
                hover_data=[
                    "StintLengths",
                    "EstimatedPitLossSeconds",
                ],
                title="Strategy scenario comparison",
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Strategy robustness analysis")
            st.caption(
                "Tests the same shortlist across 9 combinations: "
                "pit loss of 18/23/28 seconds and degradation at "
                "75%/100%/125% of your selected assumptions."
            )

            if st.button(
                "Run robustness analysis",
                key="run_strategy_robustness",
            ):
                try:
                    robustness_config = StrategyConfig(
                        total_laps=int(total_laps),
                        pit_loss_seconds=float(pit_loss),
                        min_stint_laps=int(min_stint),
                        max_stops=int(max_stops),
                        compound_pace_delta=tuple(
                            pace_inputs.items()
                        ),
                        compound_degradation=tuple(
                            degradation_inputs.items()
                        ),
                    )

                    robust_results = evaluate_candidate_robustness(
                        candidates=candidates,
                        base_config=robustness_config,
                    )

                    st.session_state["strategy_robustness_results"] = (
                        robust_results
                    )
                    st.session_state["strategy_robustness_event"] = event

                except Exception as exc:
                    st.error(
                        f"Robustness analysis failed: {exc}"
                    )

            robust_results = st.session_state.get(
                "strategy_robustness_results", pd.DataFrame()
            )

            if (
                not robust_results.empty
                and st.session_state.get(
                    "strategy_robustness_event"
                ) == event
            ):
                st.markdown("##### Most robust candidates")

                st.dataframe(
                    robust_results.head(10),
                    use_container_width=True,
                    hide_index=True,
                )

                best_robust = robust_results.iloc[0]

                st.markdown("##### Robustness leader")
                st.write(
                    f"**{best_robust['CompoundSequence']}** — "
                    f"{best_robust['StintLengths']} laps per stint, "
                    f"{int(best_robust['Stops'])} stop(s)."
                )

                st.write(
                    f"Fractional win rate: "
                    f"{best_robust['FractionalWinRate']:.1%} · "
                    f"Mean regret: "
                    f"{best_robust['MeanRegretSeconds']:.2f} seconds · "
                    f"Worst-case regret: "
                    f"{best_robust['WorstCaseRegretSeconds']:.2f} seconds."
                )

                st.caption(
                    "Fractional win rate shares credit between tied "
                    "winners. Regret is the time gap versus the best "
                    "candidate in each tested scenario. Results apply "
                    "only to this candidate shortlist and these assumptions."
                )


            st.warning(
                "Scenario scores are calculated from the selected "
                "assumptions and a neutral 0-second baseline pace. "
                "They are not validated race-time predictions. "
                "Driver-specific pace, fuel, traffic, weather, safety cars, "
                "track position and tyre allocation rules are not modelled. "
                "Pit windows are sampled in two-lap steps."
            )

        else:
            st.info(
                "Adjust the assumptions and click 'Generate strategy "
                "candidates' to compare strategies."
            )
