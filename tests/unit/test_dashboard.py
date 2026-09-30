import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from jev_experiment.benchmark import BenchmarkPlan, THROUGHPUT_MODE
from jev_experiment.dashboard import (
    _downloadable_report,
    _load_saved_job,
    _provider_statistics,
    _run_settings_from_job,
)
from jev_experiment.models import (
    CaseEvaluation,
    CaseRecord,
    CaseResult,
    DatasetSplit,
    Ownership,
    Prediction,
    ReviewStatus,
    Usage,
)


def test_load_saved_job_restores_provider_progress_and_cost(tmp_path: Path) -> None:
    case = CaseRecord(
        id="TEST-0001",
        case_text="A fictional support request.",
        gold_label=Ownership.MINE,
        reason_code="FIXTURE",
        scenario_family="fixture",
        split=DatasetSplit.DEVELOPMENT,
        difficulty="easy",
        review_status=ReviewStatus.APPROVED,
    )
    providers = ("jev", "haiku", "sonnet")
    plan = BenchmarkPlan(
        dataset_path=Path("fixture.csv"),
        cases=(case,),
        models={provider: f"{provider}-fixture" for provider in providers},
        repetitions=1,
        planned_calls=3,
        estimate_source=Path("costs.json"),
        estimated_cost_by_provider={provider: 0.001 for provider in providers},
        estimated_cost_per_case=0.003,
        estimated_cost_usd=0.003,
        retry_reserve=0.25,
        guarded_estimate_usd=0.00375,
    )
    predictions = {
        provider: Prediction(
            provider=provider,
            requested_model=f"{provider}-fixture",
            label=Ownership.MINE,
            latency_ms=10,
            usage=Usage(cost_usd=0.001),
        )
        for provider in providers
    }
    result = CaseResult(
        case_id=case.id,
        expected_label=case.gold_label,
        predictions=predictions,
        evaluation=CaseEvaluation(
            correct_by_provider={provider: True for provider in providers},
            disagreement=False,
        ),
        created_at=datetime.now(UTC),
    )
    saved = tmp_path / "benchmark-throughput-fixture.json"
    saved.write_text(
        json.dumps([result.model_dump(mode="json")]),
        encoding="utf-8",
    )

    job = _load_saved_job(plan, saved)

    assert job.done_event.is_set()
    assert job.execution_mode == THROUGHPUT_MODE
    assert job.maximum_cost_usd is None
    assert job.provider_completed_calls == {provider: 1 for provider in providers}
    assert job.provider_cost_usd == pytest.approx(
        {provider: 0.001 for provider in providers}
    )
    assert job.actual_cost_usd == pytest.approx(0.003)
    assert job.execution is not None
    assert job.execution.stopped_reason is None
    assert _run_settings_from_job(job) == {
        "dashboard_dataset": "fixture.csv",
        "dashboard_repetitions": 1,
        "dashboard_execution_mode": "Throughput",
    }
    refreshed_report = _downloadable_report(job)
    assert "# Case Ownership Routing Report" in refreshed_report
    assert "## Per-label quality" in refreshed_report
    assert "## Confusion matrices" in refreshed_report
    accuracy, macro_f1, _latency, _completed, _failures = _provider_statistics(
        job, "jev"
    )
    assert accuracy == "100.0%"
    assert macro_f1 == "33.3%"
