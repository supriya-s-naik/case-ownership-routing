"""Deterministic classification metrics for ownership-routing results."""

from collections.abc import Iterable
from dataclasses import dataclass

from jev_experiment.models import Ownership


LABELS = tuple(Ownership)


@dataclass(frozen=True)
class LabelMetrics:
    """One-vs-rest quality metrics for a single ownership label."""

    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ClassificationMetrics:
    """Three-class quality metrics calculated over successful predictions."""

    accuracy: float | None
    macro_f1: float | None
    per_label: dict[Ownership, LabelMetrics]
    confusion_matrix: dict[Ownership, dict[Ownership, int]]
    sample_count: int


def classification_metrics(
    outcomes: Iterable[tuple[Ownership, Ownership]],
) -> ClassificationMetrics:
    """Calculate accuracy, Macro F1, per-label metrics, and a confusion matrix.

    Rows in the confusion matrix are gold labels and columns are predictions.
    Undefined precision, recall, or F1 values use zero, matching the conservative
    convention for a model that never predicts or never encounters a label.
    """

    collected = list(outcomes)
    matrix = {
        expected: {predicted: 0 for predicted in LABELS} for expected in LABELS
    }
    for expected, predicted in collected:
        matrix[expected][predicted] += 1

    per_label: dict[Ownership, LabelMetrics] = {}
    for label in LABELS:
        true_positive = matrix[label][label]
        false_positive = sum(
            matrix[expected][label] for expected in LABELS if expected != label
        )
        false_negative = sum(
            matrix[label][predicted] for predicted in LABELS if predicted != label
        )
        support = sum(matrix[label].values())
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator if precision_denominator else 0.0
        )
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_label[label] = LabelMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
        )

    sample_count = len(collected)
    correct = sum(matrix[label][label] for label in LABELS)
    return ClassificationMetrics(
        accuracy=correct / sample_count if sample_count else None,
        macro_f1=(
            sum(metrics.f1 for metrics in per_label.values()) / len(LABELS)
            if sample_count
            else None
        ),
        per_label=per_label,
        confusion_matrix=matrix,
        sample_count=sample_count,
    )
