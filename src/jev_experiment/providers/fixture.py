"""Offline provider for deterministic graph tests and local development."""

import asyncio
from dataclasses import dataclass, field

from jev_experiment.models import Ownership, Prediction, Usage
from jev_experiment.providers.base import ClassificationRequest


@dataclass
class FixtureProvider:
    """Return a fixed label, optionally failing initial attempts or responding slowly."""

    name: str
    label: Ownership
    requested_model: str = "fixture-v1"
    confidence: float = 1.0
    failures_before_success: int = 0
    delay_seconds: float = 0.0
    cost_usd: float | None = None
    requests: list[ClassificationRequest] = field(default_factory=list, init=False)
    call_count: int = field(default=0, init=False)

    async def classify(self, request: ClassificationRequest) -> Prediction:
        self.requests.append(request)
        self.call_count += 1
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.call_count <= self.failures_before_success:
            raise RuntimeError(f"{self.name} fixture failure {self.call_count}")
        return Prediction(
            provider=self.name,
            requested_model=self.requested_model,
            served_model=self.requested_model,
            label=self.label,
            confidence=self.confidence,
            latency_ms=0,
            usage=Usage(cost_usd=self.cost_usd),
        )
