"""Typed LangGraph workflow for one ownership-classification case."""

import asyncio
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from time import perf_counter
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from jev_experiment.models import (
    CaseEvaluation,
    CaseRecord,
    CaseResult,
    Ownership,
    Prediction,
    ProviderFailure,
)
from jev_experiment.persistence import ResultStore
from jev_experiment.providers.base import ClassificationRequest, ClassifierProvider


ProviderProgressCallback = Callable[
    [str, int, str, Prediction | ProviderFailure],
    None,
]


def _merge_provider_maps(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    overlap = set(left) & set(right)
    if overlap:
        raise ValueError(f"Duplicate provider result: {', '.join(sorted(overlap))}")
    return {**left, **right}


class CaseGraphInput(BaseModel):
    """Validated graph input, including evaluation-only fields."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(pattern=r"^[A-Z]+-\d{4}$")
    repetition: int = Field(default=1, ge=1)
    transcript: str = Field(min_length=1)
    expected_label: Ownership
    policy_text: str = Field(min_length=1)


class CaseState(TypedDict, total=False):
    case_id: str
    repetition: int
    transcript: str
    expected_label: Ownership
    policy_text: str
    predictions: Annotated[dict[str, Prediction], _merge_provider_maps]
    failures: Annotated[dict[str, ProviderFailure], _merge_provider_maps]
    evaluation: CaseEvaluation
    result: CaseResult
    persisted_to: str | None


def _validate_case(state: CaseState) -> CaseState:
    validated = CaseGraphInput.model_validate(
        {
            "case_id": state.get("case_id"),
            "repetition": state.get("repetition", 1),
            "transcript": state.get("transcript"),
            "expected_label": state.get("expected_label"),
            "policy_text": state.get("policy_text"),
        }
    )
    return {
        **validated.model_dump(),
        "predictions": {},
        "failures": {},
    }


async def classify_with_retry(
    provider: ClassifierProvider,
    request: ClassificationRequest,
    *,
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
) -> Prediction | ProviderFailure:
    started = perf_counter()
    for retry_count in range(max_retries + 1):
        try:
            prediction = await asyncio.wait_for(
                provider.classify(request),
                timeout=timeout_seconds,
            )
            if prediction.provider != provider.name:
                raise ValueError(
                    f"Provider returned identity {prediction.provider!r}; "
                    f"expected {provider.name!r}"
                )
            return prediction.model_copy(
                update={
                    "requested_model": provider.requested_model,
                    "retry_count": retry_count,
                    "latency_ms": (perf_counter() - started) * 1000,
                }
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if retry_count == max_retries:
                return ProviderFailure(
                    provider=provider.name,
                    requested_model=provider.requested_model,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    retry_count=retry_count,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            if retry_backoff_seconds:
                await asyncio.sleep(retry_backoff_seconds * (2**retry_count))
    raise AssertionError("retry loop exited without an outcome")


def _provider_node(
    provider: ClassifierProvider,
    *,
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
    progress_callback: ProviderProgressCallback | None = None,
):
    async def classify(state: CaseState) -> CaseState:
        request = ClassificationRequest(
            policy_text=state["policy_text"],
            case_transcript=state["transcript"],
        )
        outcome = await classify_with_retry(
            provider,
            request,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        if progress_callback is not None:
            progress_callback(
                state["case_id"],
                state.get("repetition", 1),
                provider.name,
                outcome,
            )
        if isinstance(outcome, Prediction):
            return {"predictions": {provider.name: outcome}}
        return {"failures": {provider.name: outcome}}

    classify.__name__ = f"classify_{provider.name}"
    return classify


def _score_case(state: CaseState) -> CaseState:
    predictions = dict(sorted(state.get("predictions", {}).items()))
    return {
        "evaluation": CaseEvaluation(
            correct_by_provider={
                provider: prediction.label == state["expected_label"]
                for provider, prediction in predictions.items()
            },
            disagreement=False,
        ),
    }


def _analyze_case(
    state: CaseState,
    *,
    low_confidence_threshold: float,
) -> CaseState:
    predictions = state.get("predictions", {})
    failures = state.get("failures", {})
    labels = {prediction.label for prediction in predictions.values()}
    evaluation = state["evaluation"].model_copy(
        update={
            "disagreement": len(labels) > 1,
            "low_confidence_providers": sorted(
                provider
                for provider, prediction in predictions.items()
                if prediction.confidence is not None
                and prediction.confidence < low_confidence_threshold
            ),
            "failure_providers": sorted(failures),
        }
    )
    return {
        "evaluation": evaluation,
        "result": CaseResult(
            case_id=state["case_id"],
            repetition=state.get("repetition", 1),
            expected_label=state["expected_label"],
            predictions=dict(sorted(predictions.items())),
            failures=dict(sorted(failures.items())),
            evaluation=evaluation,
            created_at=datetime.now(UTC),
        ),
    }


def _persistence_node(store: ResultStore | None):
    async def persist(state: CaseState) -> CaseState:
        persisted_to = await store.save(state["result"]) if store is not None else None
        return {"persisted_to": persisted_to}

    return persist


def build_case_graph(
    providers: Sequence[ClassifierProvider],
    *,
    store: ResultStore | None = None,
    timeout_seconds: float = 30,
    max_retries: int = 2,
    retry_backoff_seconds: float = 0.25,
    low_confidence_threshold: float = 0.5,
    provider_progress_callback: ProviderProgressCallback | None = None,
):
    """Compile one graph with an independent branch for every enabled provider."""

    if not providers:
        raise ValueError("At least one provider must be enabled")
    names = [provider.name for provider in providers]
    if len(names) != len(set(names)):
        raise ValueError("Provider names must be unique")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    if retry_backoff_seconds < 0:
        raise ValueError("retry_backoff_seconds cannot be negative")
    if not 0 <= low_confidence_threshold <= 1:
        raise ValueError("low_confidence_threshold must be between 0 and 1")

    builder = StateGraph(CaseState)
    builder.add_node("validate_case", _validate_case)
    provider_nodes: list[str] = []
    for provider in providers:
        node_name = f"classify_{provider.name}"
        provider_nodes.append(node_name)
        builder.add_node(
            node_name,
            _provider_node(
                provider,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                progress_callback=provider_progress_callback,
            ),
        )
        builder.add_edge("validate_case", node_name)

    def analyze(state: CaseState) -> CaseState:
        return _analyze_case(
            state,
            low_confidence_threshold=low_confidence_threshold,
        )

    builder.add_node("score_case", _score_case)
    builder.add_node("analyze_case", analyze)
    builder.add_node("persist_result", _persistence_node(store))
    builder.add_edge(START, "validate_case")
    builder.add_edge(provider_nodes, "score_case")
    builder.add_edge("score_case", "analyze_case")
    builder.add_edge("analyze_case", "persist_result")
    builder.add_edge("persist_result", END)
    return builder.compile(name="ownership_case")


async def run_case_graph(
    graph,
    case: CaseRecord,
    *,
    policy_text: str,
    repetition: int = 1,
) -> CaseResult:
    """Run one validated dataset record and return its complete normalized result."""

    final_state = await graph.ainvoke(
        {
            "case_id": case.id,
            "repetition": repetition,
            "transcript": case.case_text,
            "expected_label": case.gold_label,
            "policy_text": policy_text,
        }
    )
    return CaseResult.model_validate(final_state["result"])
