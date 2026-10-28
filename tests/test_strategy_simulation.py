
import pytest

from pitwall.analytics.pre_race import (
    StrategyConfig,
    generate_candidate_strategies,
    simulate_strategy,
)


@pytest.fixture
def config():
    return StrategyConfig(
        total_laps=56,
        pit_loss_seconds=23.0,
        min_stint_laps=5,
        max_stops=3,
    )


def test_one_stop_pit_loss(config):
    result = simulate_strategy(
        ("SOFT", "MEDIUM"),
        (24, 32),
        102.0,
        config,
    )

    assert result["Stops"] == 1
    assert result["EstimatedPitLossSeconds"] == 23.0


def test_safety_car_discount_reduces_pit_loss(config):
    result = simulate_strategy(
        ("SOFT", "MEDIUM"),
        (24, 32),
        102.0,
        config,
        safety_car_pit_discount=10.0,
    )

    assert result["EstimatedPitLossSeconds"] == 13.0


def test_stint_lengths_must_equal_race_distance(config):
    with pytest.raises(ValueError):
        simulate_strategy(
            ("SOFT", "MEDIUM"),
            (24, 30),
            102.0,
            config,
        )


def test_candidate_strategies_obey_constraints(config):
    candidates = generate_candidate_strategies(
        driver_pace_seconds=102.0,
        config=config,
        pit_window_step=5,
    )

    assert not candidates.empty
    assert set(candidates["Stops"]).issubset({1, 2, 3})

    for row in candidates.itertuples(index=False):
        lengths = [
            int(length)
            for length in row.StintLengths.split(" / ")
        ]
        compounds = row.CompoundSequence.split(" -> ")

        assert sum(lengths) == config.total_laps
        assert all(
            length >= config.min_stint_laps
            for length in lengths
        )
        assert len(compounds) == row.Stops + 1
        assert all(
            compounds[i] != compounds[i + 1]
            for i in range(len(compounds) - 1)
        )
        assert row.EstimatedPitLossSeconds == (
            row.Stops * config.pit_loss_seconds
        )
