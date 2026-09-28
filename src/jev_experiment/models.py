"""Typed records shared by datasets, providers, and evaluation."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Ownership(StrEnum):
    MINE = "MINE"
    NOT_MINE = "NOT_MINE"
    UNSURE = "UNSURE"


class DatasetSplit(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    ROBUSTNESS = "robustness"


class ReviewStatus(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"


class CaseRecord(BaseModel):
    """A local labeled case. Only ``case_text`` is model-facing."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Z]+-\d{4}$")
    case_text: str = Field(min_length=1)
    gold_label: Ownership
    reason_code: str = Field(min_length=1)
    scenario_family: str = Field(min_length=1)
    split: DatasetSplit
    difficulty: str = Field(min_length=1)
    pair_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.DRAFT
    review_notes: str | None = None


class Usage(BaseModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)


class Prediction(BaseModel):
    provider: str
    requested_model: str
    served_model: str | None = None
    label: Ownership
    probabilities: dict[Ownership, float] | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    latency_ms: float = Field(ge=0)
    retry_count: int = Field(default=0, ge=0)
    usage: Usage = Field(default_factory=Usage)
    raw_response: dict[str, object] | None = None

    @model_validator(mode="after")
    def validate_probabilities(self) -> "Prediction":
        if self.probabilities is None:
            return self
        if set(self.probabilities) != set(Ownership):
            raise ValueError("probabilities must contain all ownership labels")
        if any(value < 0 or value > 1 for value in self.probabilities.values()):
            raise ValueError("probabilities must be between 0 and 1")
        return self


class ProviderFailure(BaseModel):
    provider: str
    requested_model: str
    error_type: str
    message: str
    retry_count: int = Field(default=0, ge=0)
    latency_ms: float = Field(ge=0)


class CaseEvaluation(BaseModel):
    correct_by_provider: dict[str, bool]
    disagreement: bool
    low_confidence_providers: list[str] = Field(default_factory=list)
    failure_providers: list[str] = Field(default_factory=list)


class CaseResult(BaseModel):
    case_id: str
    repetition: int = Field(default=1, ge=1)
    expected_label: Ownership
    predictions: dict[str, Prediction] = Field(default_factory=dict)
    failures: dict[str, ProviderFailure] = Field(default_factory=dict)
    evaluation: CaseEvaluation | None = None
    created_at: datetime
