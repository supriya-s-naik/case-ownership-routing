"""Jev adapter using the TypeSafe SDK against OpenRouter."""

from time import perf_counter
from typing import Any

from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from jev_experiment.models import Ownership, Prediction, Usage
from jev_experiment.providers.contract import QUESTION_ID, decision_state, ownership_questions

OPENROUTER_SYSTEM_ONE_BASE_URL = "https://openrouter.ai/api"


def _reported_cost(raw_payload: object) -> float | None:
    """Extract OpenRouter's provider-reported cost without estimating it."""

    if not isinstance(raw_payload, dict):
        return None
    usage = raw_payload.get("usage")
    if not isinstance(usage, dict):
        return None
    cost = usage.get("cost")
    return float(cost) if isinstance(cost, (int, float)) else None


def _raw_response_record(response: Any) -> tuple[dict[str, object], float | None]:
    """Retain fields the TypeSafe response schema may not model, including cost."""

    raw_http_response = response.raw_http_response
    raw_payload = raw_http_response.json()
    record: dict[str, object] = {
        "body": raw_payload,
        "metadata": {
            "status_code": raw_http_response.status_code,
            "generation_id": raw_http_response.headers.get("x-generation-id"),
            "request_id": raw_http_response.headers.get("x-typesafe-request-id"),
        },
        "normalized": response.model_dump(mode="json"),
    }
    return record, _reported_cost(raw_payload)


async def classify_with_jev(
    *,
    case_text: str,
    policy_text: str,
    api_key: str,
    model: str,
    timeout_seconds: float,
    max_retries: int,
) -> Prediction:
    """Classify one transcript with pinned Jev through OpenRouter."""

    started = perf_counter()
    retry = RetryPolicy(max_retries=max_retries, timeout=timeout_seconds)
    async with AsyncTypeSafeClient(
        api_key=api_key,
        base_url=OPENROUTER_SYSTEM_ONE_BASE_URL,
        retry=retry,
        timeout=timeout_seconds,
    ) as client:
        response = await client.system_one(
            model=model.removeprefix("typesafe/"),
            state=decision_state(case_text, policy_text),
            questions=ownership_questions(),
        )

    answer = response.choices[QUESTION_ID]
    raw_response, cost_usd = _raw_response_record(response)
    return Prediction(
        provider="jev",
        requested_model=model,
        served_model=response.model,
        label=Ownership(answer.choice),
        probabilities={Ownership(key): value for key, value in answer.probabilities.items()},
        confidence=answer.confidence,
        latency_ms=(perf_counter() - started) * 1000,
        usage=Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=cost_usd,
        ),
        raw_response=raw_response,
    )
