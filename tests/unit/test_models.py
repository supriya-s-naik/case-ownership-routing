import pytest
from pydantic import ValidationError

from jev_experiment.models import Ownership, Prediction


def test_prediction_requires_every_probability() -> None:
    with pytest.raises(ValidationError, match="all ownership labels"):
        Prediction(
            provider="fixture",
            requested_model="fixture-v1",
            label=Ownership.MINE,
            probabilities={Ownership.MINE: 1.0},
            latency_ms=1,
        )
