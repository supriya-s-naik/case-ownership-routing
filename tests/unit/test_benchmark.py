import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from jev_experiment.benchmark import (
    BenchmarkPlan,
    BenchmarkProgress,
    THROUGHPUT_MODE,
    execute_benchmark,
    prepare_benchmark_plan,
)
from jev_experiment.dataset import load_cases
from jev_experiment.models import (
    CaseEvaluation,
    CaseResult,
    Ownership,
    Prediction,
    ReviewStatus,
    Usage,
)
from jev_experiment.providers.fixture import FixtureProvider


def write_cost_source(path: Path) -> None:
    predictions = {
        "jev": Prediction(
            provider="jev",
            requested_model="jev-fixture",
            label=Ownership.MINE,
            latency_ms=1,
            usage=Usage(cost_usd=0.001),
        ),
        "haiku": Prediction(
            provider="haiku",
            requested_model="haiku-fixture",
            label=Ownership.MINE,
            latency_ms=1,
            usage=Usage(cost_usd=0.002),
        ),
        "sonnet": Prediction(
            provider="sonnet",
            requested_model="sonnet-fixture",
            label=Ownership.MINE,
            latency_ms=1,
            usage=Usage(cost_usd=0.003),
        ),
    }
    result = CaseResult(
        case_id="TEST-0001",
        expected_label=Ownership.MINE,
        predictions=predictions,
        evaluation=CaseEvaluation(
            correct_by_provider={provider: True for provider in predictions},
            disagreement=False,
        ),
        created_at=datetime.now(UTC),
    )
    path.write_text(
        json.dumps([result.model_dump(mode="json")]),
        encoding="utf-8",
    )


def test_benchmark_plan_counts_calls_and_adds_retry_reserve(tmp_path: Path) -> None:
    cost_source = tmp_path / "costs.json"
    write_cost_source(cost_source)

    plan = prepare_benchmark_plan(
        Path("data/seeds/cases.csv"),
        repetitions=2,
        estimate_from=cost_source,
    )

    assert len(plan.cases) == 26
    assert plan.planned_calls == 26 * 3 * 2
    assert plan.estimated_cost_per_case == pytest.approx(0.006)
    assert plan.estimated_cost_usd == pytest.approx(0.312)
    assert plan.guarded_estimate_usd == pytest.approx(0.390)


def test_benchmark_plan_rejects_draft_cases(tmp_path: Path) -> None:
    cost_source = tmp_path / "costs.json"
    write_cost_source(cost_source)
    draft = load_cases(Path("data/seeds/cases.csv"))[0].model_copy(
        update={"review_status": ReviewStatus.DRAFT}
    )
    dataset = tmp_path / "draft.jsonl"
    dataset.write_text(draft.model_dump_json() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires approved cases"):
        prepare_benchmark_plan(dataset, estimate_from=cost_source)


def execution_plan() -> BenchmarkPlan:
    cases = tuple(load_cases(Path("data/development.jsonl"))[:2])
    estimates = {"jev": 0.002, "haiku": 0.002, "sonnet": 0.002}
    return BenchmarkPlan(
        dataset_path=Path("data/development.jsonl"),
        cases=cases,
        models={provider: f"{provider}-fixture" for provider in estimates},
        repetitions=1,
        planned_calls=6,
        estimate_source=Path("fixture-costs.json"),
        estimated_cost_by_provider=estimates,
        estimated_cost_per_case=0.006,
        estimated_cost_usd=0.012,
        retry_reserve=0.25,
        guarded_estimate_usd=0.015,
    )


def costly_providers() -> tuple[FixtureProvider, ...]:
    return tuple(
        FixtureProvider(provider, Ownership.MINE, cost_usd=0.005)
        for provider in ("jev", "haiku", "sonnet")
    )


@pytest.mark.asyncio
async def test_benchmark_refuses_before_calls_when_cap_is_too_low(
    tmp_path: Path,
) -> None:
    providers = costly_providers()
    output = tmp_path / "blocked.json"

    with pytest.raises(ValueError, match="no API calls were made"):
        await execute_benchmark(
            execution_plan(),
            providers,
            policy_text="policy",
            maximum_cost_usd=0.01,
            output_path=output,
            timeout_seconds=1,
            max_retries=0,
            retry_backoff_seconds=0,
            low_confidence_threshold=0.5,
        )

    assert all(provider.call_count == 0 for provider in providers)
    assert not output.exists()


@pytest.mark.asyncio
async def test_benchmark_stops_before_next_case_and_keeps_partial_results(
    tmp_path: Path,
) -> None:
    providers = costly_providers()
    output = tmp_path / "partial.json"

    execution = await execute_benchmark(
        execution_plan(),
        providers,
        policy_text="policy",
        maximum_cost_usd=0.02,
        output_path=output,
        timeout_seconds=1,
        max_retries=0,
        retry_backoff_seconds=0,
        low_confidence_threshold=0.5,
    )

    assert len(execution.results) == 1
    assert execution.actual_reported_cost_usd == pytest.approx(0.015)
    assert execution.stopped_reason is not None
    assert "next case" in execution.stopped_reason
    assert output.exists()
    assert execution.report_path.exists()
    assert len(json.loads(output.read_text(encoding="utf-8"))) == 1
    assert all(provider.call_count == 1 for provider in providers)


@pytest.mark.asyncio
async def test_dashboard_progress_can_request_stop_after_current_case(
    tmp_path: Path,
) -> None:
    providers = costly_providers()
    output = tmp_path / "user-stopped.json"
    events: list[BenchmarkProgress] = []

    execution = await execute_benchmark(
        execution_plan(),
        providers,
        policy_text="policy",
        maximum_cost_usd=0.05,
        output_path=output,
        timeout_seconds=1,
        max_retries=0,
        retry_backoff_seconds=0,
        low_confidence_threshold=0.5,
        progress_callback=events.append,
        should_stop=lambda: any(event.stage == "completed" for event in events),
    )

    assert events[0].stage == "running"
    assert events[-1].stage == "completed"
    provider_events = [
        event for event in events if event.stage == "provider_completed"
    ]
    assert {event.provider for event in provider_events} == {
        "jev",
        "haiku",
        "sonnet",
    }
    assert all(event.provider_completed_calls == 1 for event in provider_events)
    assert all(event.prediction is not None for event in provider_events)
    assert sorted(event.actual_reported_cost_usd for event in provider_events) == [
        pytest.approx(0.005),
        pytest.approx(0.010),
        pytest.approx(0.015),
    ]
    assert events[-1].completed_case_runs == 1
    assert events[-1].result is not None
    assert len(execution.results) == 1
    assert execution.stopped_reason == "User requested stop after the completed case"


@pytest.mark.asyncio
async def test_throughput_mode_allows_fast_provider_to_advance_independently(
    tmp_path: Path,
) -> None:
    providers = (
        FixtureProvider("jev", Ownership.MINE, cost_usd=0.001),
        FixtureProvider("haiku", Ownership.MINE, delay_seconds=0.03, cost_usd=0.001),
        FixtureProvider("sonnet", Ownership.MINE, delay_seconds=0.03, cost_usd=0.001),
    )
    events: list[BenchmarkProgress] = []

    execution = await execute_benchmark(
        execution_plan(),
        providers,
        policy_text="policy",
        maximum_cost_usd=0.05,
        output_path=tmp_path / "throughput.json",
        timeout_seconds=1,
        max_retries=0,
        retry_backoff_seconds=0,
        low_confidence_threshold=0.5,
        progress_callback=events.append,
        execution_mode=THROUGHPUT_MODE,
    )

    provider_events = [
        event for event in events if event.stage == "provider_completed"
    ]
    second_jev = next(
        index
        for index, event in enumerate(provider_events)
        if event.provider == "jev" and event.provider_completed_calls == 2
    )
    first_haiku = next(
        index
        for index, event in enumerate(provider_events)
        if event.provider == "haiku" and event.provider_completed_calls == 1
    )
    assert second_jev < first_haiku
    assert len(execution.results) == 2
    assert all(
        set(result.predictions) == {"jev", "haiku", "sonnet"}
        for result in execution.results
    )
    assert execution.stopped_reason is None
    assert "Execution mode: throughput" in execution.report_path.read_text(
        encoding="utf-8"
    )
