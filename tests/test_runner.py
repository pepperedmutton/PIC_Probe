import numpy as np

from run_physics_accurate import main, make_seed_plan, summarize_current


def test_seed_plan_has_independent_directions() -> None:
    plan = make_seed_plan(42, 4)
    assert len(plan) == 8
    assert len({item["seed"] for item in plan}) == 8
    assert {item["direction"] for item in plan} == {"up", "down"}


def test_current_summary_contains_the_mean() -> None:
    values = np.linspace(0.9, 1.1, 100)
    summary = summarize_current(values)
    assert summary.confidence_low_a < summary.mean_a < summary.confidence_high_a
    assert summary.sample_count == 100


def test_plan_only_cli_does_not_run_simulation(capsys) -> None:
    result = main(["--plan-only", "--points", "3", "--repeats", "1"])
    output = capsys.readouterr().out
    assert result == 0
    assert "seed_plan" in output
