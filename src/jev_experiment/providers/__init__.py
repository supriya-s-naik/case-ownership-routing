"""Provider adapters for the ownership benchmark."""

from jev_experiment.providers.base import (
    CallableProvider,
    ClassificationRequest,
    ClassifierProvider,
)
from jev_experiment.providers.fixture import FixtureProvider

__all__ = [
    "CallableProvider",
    "ClassificationRequest",
    "ClassifierProvider",
    "FixtureProvider",
]
