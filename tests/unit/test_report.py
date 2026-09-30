from datetime import UTC, datetime

from jev_experiment.evaluation import render_smoke_report
from jev_experiment.models import (
    CaseEvaluation,
    CaseResult,
    Ownership,
    Prediction,
    Usage,
)


def test_smoke_report_summarizes_predictions() -> None:
    prediction = Prediction(
        provider="jev",
        requested_model="typesafe/jev-1.13",
        served_model="jev-1.13",
        label=Ownership.MINE,
        confidence=0.9,
        latency_ms=125,
        usage=Usage(input_tokens=20, output_tokens=3),
    )
    result = CaseResult(
        case_id="DEV-0001",
        expected_label=Ownership.MINE,
        predictions={"jev": prediction},
        evaluation=CaseEvaluation(correct_by_provider={"jev": True}, disagreement=False),
        created_at=datetime.now(UTC),
    )

    report = render_smoke_report([result])

    assert (
        "| jev | 1 | 0 | 100.0% | 0.333 | 125.0 ms | 0.900 | 20 | 3 | n/a |"
        in report
    )
    assert "## Per-label quality" in report
    assert "| jev | MINE | 1.000 | 1.000 | 1.000 | 1 |" in report
    assert "## Confusion matrices" in report
    assert "| MINE | 1 | 0 | 0 |" in report
    assert "| DEV-0001 | MINE | MINE | FAILURE | FAILURE | no |" in report
    assert "Cost coverage: 0 of 1 successful calls; missing for: jev." in report


def test_smoke_report_marks_complete_cost_coverage() -> None:
    prediction = Prediction(
        provider="jev",
        requested_model="typesafe/jev-1.13",
        label=Ownership.MINE,
        confidence=1.0,
        latency_ms=100,
        usage=Usage(input_tokens=20, output_tokens=3, cost_usd=0.000001),
    )
    result = CaseResult(
        case_id="DEV-0001",
        expected_label=Ownership.MINE,
        predictions={"jev": prediction},
        evaluation=CaseEvaluation(correct_by_provider={"jev": True}, disagreement=False),
        created_at=datetime.now(UTC),
    )

    report = render_smoke_report([result])

    assert "Cost coverage: 1 of 1 successful calls (complete)." in report


def test_smoke_report_shows_case_level_confidence_and_probabilities() -> None:
    prediction = Prediction(
        provider="jev",
        requested_model="typesafe/jev-1.13",
        label=Ownership.UNSURE,
        probabilities={
            Ownership.MINE: 0.01,
            Ownership.NOT_MINE: 0.00,
            Ownership.UNSURE: 0.99,
        },
        confidence=0.98,
        latency_ms=100,
    )
    result = CaseResult(
        case_id="DEV-0003",
        expected_label=Ownership.UNSURE,
        predictions={"jev": prediction},
        evaluation=CaseEvaluation(
            correct_by_provider={"jev": True},
            disagreement=False,
            low_confidence_providers=[],
        ),
        created_at=datetime.now(UTC),
    )

    report = render_smoke_report([result])

    assert "## Case confidence and probabilities" in report
    assert (
        "| DEV-0003 | jev | UNSURE | 0.980 | 0.010 | 0.000 | 0.990 | no |"
        in report
    )
    assert "| DEV-0003 | haiku | FAILURE | n/a | n/a | n/a | n/a | n/a |" in report
