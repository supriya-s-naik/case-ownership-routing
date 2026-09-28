"""Provider interface used by the LangGraph workflow."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from jev_experiment.models import Prediction


class ClassificationRequest(BaseModel):
    """The complete model-facing input; evaluation fields are intentionally absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_text: str = Field(min_length=1)
    case_transcript: str = Field(min_length=1)


class ClassifierProvider(Protocol):
    """A named asynchronous classifier used by one graph branch."""

    name: str
    requested_model: str

    async def classify(self, request: ClassificationRequest) -> Prediction: ...


@dataclass(frozen=True)
class CallableProvider:
    """Adapt an async callable to the graph provider interface."""

    name: str
    requested_model: str
    callback: Callable[[ClassificationRequest], Awaitable[Prediction]]

    async def classify(self, request: ClassificationRequest) -> Prediction:
        return await self.callback(request)
