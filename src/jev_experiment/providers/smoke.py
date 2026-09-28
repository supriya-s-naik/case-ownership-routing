"""Parallel provider fan-out for the pre-LangGraph API spike."""

import asyncio
from collections.abc import Awaitable
from datetime import UTC, datetime
from time import perf_counter

from jev_experiment.models import CaseEvaluation, CaseRecord, CaseResult, Prediction, ProviderFailure
from jev_experiment.providers.claude import classify_with_claude
from jev_experiment.providers.jev import classify_with_jev


async def _capture(
    provider_name: str,
    requested_model: str,
    request: Awaitable[Prediction],
) -> Prediction | ProviderFailure:
    started = perf_counter()
    try:
        return await request
    except Exception as exc:  # provider exceptions are persisted, not converted to UNSURE
        return ProviderFailure(
            provider=provider_name,
            requested_model=requested_model,
            error_type=type(exc).__name__,
            message=str(exc),
            latency_ms=(perf_counter() - started) * 1000,
        )


async def run_smoke_case(
    case: CaseRecord,
    *,
    policy_text: str,
    api_key: str,
    models: dict[str, str],
    timeout_seconds: float,
    max_retries: int,
) -> CaseResult:
    """Run all three independent providers for one development case."""

    requests = {
        "jev": classify_with_jev(
            case_text=case.case_text,
            policy_text=policy_text,
            api_key=api_key,
            model=models["jev"],
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        ),
        "haiku": classify_with_claude(
            provider_name="haiku",
            case_text=case.case_text,
            policy_text=policy_text,
            api_key=api_key,
            model=models["haiku"],
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        ),
        "sonnet": classify_with_claude(
            provider_name="sonnet",
            case_text=case.case_text,
            policy_text=policy_text,
            api_key=api_key,
            model=models["sonnet"],
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        ),
    }
    outcomes = await asyncio.gather(
        *(
            _capture(provider, models[provider], request)
            for provider, request in requests.items()
        )
    )

    predictions = {
        outcome.provider: outcome for outcome in outcomes if isinstance(outcome, Prediction)
    }
    failures = {
        outcome.provider: outcome for outcome in outcomes if isinstance(outcome, ProviderFailure)
    }
    labels = {prediction.label for prediction in predictions.values()}
    evaluation = CaseEvaluation(
        correct_by_provider={
            provider: prediction.label == case.gold_label
            for provider, prediction in predictions.items()
        },
        disagreement=len(labels) > 1,
    )
    return CaseResult(
        case_id=case.id,
        expected_label=case.gold_label,
        predictions=predictions,
        failures=failures,
        evaluation=evaluation,
        created_at=datetime.now(UTC),
    )
