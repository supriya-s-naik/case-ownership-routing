import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import ValidationError

from jev_experiment.dataset import load_cases
from jev_experiment.graph import build_case_graph, run_case_graph
from jev_experiment.models import (
    CaseRecord,
    DatasetSplit,
    Ownership,
    Prediction,
    ReviewStatus,
)
from jev_experiment.persistence import InMemoryResultStore, JsonResultStore
from jev_experiment.providers.base import ClassificationRequest
from jev_experiment.providers.fixture import FixtureProvider


def development_case(label: Ownership = Ownership.MINE) -> CaseRecord:
    return CaseRecord(
        id="TEST-0001",
        case_text="A fictional application request fails with a confirmed exception.",
        gold_label=label,
        reason_code="FIXTURE",
        scenario_family="fixture",
        split=DatasetSplit.DEVELOPMENT,
        difficulty="easy",
        review_status=ReviewStatus.APPROVED,
    )


@pytest.mark.asyncio
async def test_graph_joins_every_enabled_provider_before_scoring() -> None:
    store = InMemoryResultStore()
    providers = (
        FixtureProvider("jev", Ownership.MINE),
        FixtureProvider("haiku", Ownership.NOT_MINE, confidence=0.4),
        FixtureProvider("sonnet", Ownership.MINE),
    )
    graph = build_case_graph(
        providers,
        store=store,
        retry_backoff_seconds=0,
    )

    result = await run_case_graph(graph, development_case(), policy_text="policy")

    assert set(result.predictions) == {"jev", "haiku", "sonnet"}
    assert result.failures == {}
    assert result.evaluation is not None
    assert result.evaluation.correct_by_provider == {
        "haiku": False,
        "jev": True,
        "sonnet": True,
    }
    assert result.evaluation.disagreement is True
    assert result.evaluation.low_confidence_providers == ["haiku"]
    assert store.results == [result]


@pytest.mark.asyncio
async def test_checked_in_development_case_returns_every_enabled_provider() -> None:
    case = load_cases(Path("data/development.jsonl"))[0]
    providers = tuple(
        FixtureProvider(name, case.gold_label)
        for name in ("jev", "haiku", "sonnet")
    )
    graph = build_case_graph(providers, retry_backoff_seconds=0)

    result = await run_case_graph(graph, case, policy_text="policy")

    assert set(result.predictions) == {"jev", "haiku", "sonnet"}
    assert result.evaluation is not None
    assert all(result.evaluation.correct_by_provider.values())


@pytest.mark.asyncio
async def test_provider_requests_exclude_gold_and_evaluation_metadata() -> None:
    provider = FixtureProvider("fixture", Ownership.MINE)
    graph = build_case_graph((provider,), retry_backoff_seconds=0)

    await run_case_graph(graph, development_case(), policy_text="policy")

    assert len(provider.requests) == 1
    assert provider.requests[0].model_dump() == {
        "policy_text": "policy",
        "case_transcript": (
            "A fictional application request fails with a confirmed exception."
        ),
    }
    assert "gold" not in repr(provider.requests[0]).lower()


@pytest.mark.asyncio
async def test_validate_node_rejects_an_empty_transcript() -> None:
    graph = build_case_graph(
        (FixtureProvider("fixture", Ownership.MINE),),
        retry_backoff_seconds=0,
    )

    with pytest.raises(ValidationError):
        await graph.ainvoke(
            {
                "case_id": "TEST-0001",
                "transcript": "",
                "expected_label": Ownership.MINE,
                "policy_text": "policy",
            }
        )


@pytest.mark.asyncio
async def test_graph_retries_a_transient_provider_failure() -> None:
    provider = FixtureProvider(
        "fixture",
        Ownership.MINE,
        failures_before_success=1,
    )
    graph = build_case_graph(
        (provider,),
        max_retries=2,
        retry_backoff_seconds=0,
    )

    result = await run_case_graph(graph, development_case(), policy_text="policy")

    assert provider.call_count == 2
    assert result.predictions["fixture"].retry_count == 1
    assert result.failures == {}


@pytest.mark.asyncio
async def test_graph_captures_exhausted_failure_without_inventing_unsure() -> None:
    provider = FixtureProvider(
        "fixture",
        Ownership.UNSURE,
        failures_before_success=10,
    )
    graph = build_case_graph(
        (provider,),
        max_retries=1,
        retry_backoff_seconds=0,
    )

    result = await run_case_graph(graph, development_case(), policy_text="policy")

    assert result.predictions == {}
    assert result.failures["fixture"].error_type == "RuntimeError"
    assert result.failures["fixture"].retry_count == 1
    assert result.evaluation is not None
    assert result.evaluation.correct_by_provider == {}
    assert result.evaluation.failure_providers == ["fixture"]


@pytest.mark.asyncio
async def test_graph_captures_provider_timeout() -> None:
    provider = FixtureProvider(
        "fixture",
        Ownership.MINE,
        delay_seconds=0.05,
    )
    graph = build_case_graph(
        (provider,),
        timeout_seconds=0.005,
        max_retries=0,
        retry_backoff_seconds=0,
    )

    result = await run_case_graph(graph, development_case(), policy_text="policy")

    assert result.predictions == {}
    assert result.failures["fixture"].error_type == "TimeoutError"


@pytest.mark.asyncio
async def test_json_store_persists_a_complete_result(tmp_path: Path) -> None:
    output = tmp_path / "run.json"
    store = JsonResultStore(output)
    graph = build_case_graph(
        (FixtureProvider("fixture", Ownership.MINE),),
        store=store,
        retry_backoff_seconds=0,
    )

    result = await run_case_graph(graph, development_case(), policy_text="policy")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert len(payload) == 1
    assert payload[0]["case_id"] == result.case_id
    assert payload[0]["predictions"]["fixture"]["label"] == "MINE"


@dataclass
class ParallelCoordinator:
    expected: int
    started: int = 0
    all_started: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class CoordinatedProvider:
    name: str
    coordinator: ParallelCoordinator
    requested_model: str = "parallel-fixture-v1"

    async def classify(self, request: ClassificationRequest) -> Prediction:
        self.coordinator.started += 1
        if self.coordinator.started == self.coordinator.expected:
            self.coordinator.all_started.set()
        await asyncio.wait_for(self.coordinator.all_started.wait(), timeout=0.2)
        return Prediction(
            provider=self.name,
            requested_model=self.requested_model,
            label=Ownership.MINE,
            latency_ms=0,
        )


@pytest.mark.asyncio
async def test_provider_nodes_fan_out_in_parallel() -> None:
    coordinator = ParallelCoordinator(expected=3)
    providers = tuple(
        CoordinatedProvider(name, coordinator)
        for name in ("jev", "haiku", "sonnet")
    )
    graph = build_case_graph(
        providers,
        timeout_seconds=0.5,
        max_retries=0,
        retry_backoff_seconds=0,
    )

    result = await asyncio.wait_for(
        run_case_graph(graph, development_case(), policy_text="policy"),
        timeout=1,
    )

    assert coordinator.started == 3
    assert set(result.predictions) == {"jev", "haiku", "sonnet"}
    assert result.failures == {}


@pytest.mark.asyncio
async def test_graph_reports_each_provider_as_soon_as_it_finishes() -> None:
    events: list[tuple[str, int, str, object]] = []
    providers = tuple(
        FixtureProvider(name, Ownership.MINE)
        for name in ("jev", "haiku", "sonnet")
    )
    graph = build_case_graph(
        providers,
        retry_backoff_seconds=0,
        provider_progress_callback=lambda case_id, repetition, provider, outcome: (
            events.append((case_id, repetition, provider, outcome))
        ),
    )

    await run_case_graph(graph, development_case(), policy_text="policy")

    assert {event[2] for event in events} == {"jev", "haiku", "sonnet"}
    assert all(event[0] == "TEST-0001" for event in events)
    assert all(event[1] == 1 for event in events)
    assert all(isinstance(event[3], Prediction) for event in events)
