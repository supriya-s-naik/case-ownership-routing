import pytest

from jev_experiment.evaluation import classification_metrics
from jev_experiment.models import Ownership


def test_classification_metrics_calculates_three_class_quality() -> None:
    metrics = classification_metrics(
        [
            (Ownership.MINE, Ownership.MINE),
            (Ownership.MINE, Ownership.NOT_MINE),
            (Ownership.NOT_MINE, Ownership.NOT_MINE),
            (Ownership.NOT_MINE, Ownership.NOT_MINE),
            (Ownership.UNSURE, Ownership.UNSURE),
            (Ownership.UNSURE, Ownership.MINE),
        ]
    )

    assert metrics.sample_count == 6
    assert metrics.accuracy == pytest.approx(4 / 6)
    assert metrics.macro_f1 == pytest.approx((0.5 + 0.8 + (2 / 3)) / 3)
    assert metrics.per_label[Ownership.MINE].precision == pytest.approx(0.5)
    assert metrics.per_label[Ownership.MINE].recall == pytest.approx(0.5)
    assert metrics.per_label[Ownership.NOT_MINE].precision == pytest.approx(2 / 3)
    assert metrics.per_label[Ownership.NOT_MINE].recall == pytest.approx(1.0)
    assert metrics.per_label[Ownership.UNSURE].support == 2
    assert metrics.confusion_matrix[Ownership.UNSURE] == {
        Ownership.MINE: 1,
        Ownership.NOT_MINE: 0,
        Ownership.UNSURE: 1,
    }


def test_classification_metrics_handles_no_predictions() -> None:
    metrics = classification_metrics([])

    assert metrics.sample_count == 0
    assert metrics.accuracy is None
    assert metrics.macro_f1 is None
    assert all(label_metrics.f1 == 0 for label_metrics in metrics.per_label.values())
