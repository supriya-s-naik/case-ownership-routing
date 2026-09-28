from collections import Counter
from pathlib import Path

from jev_experiment.dataset import (
    load_cases,
    load_reason_codes,
    validate_case_collection,
)
from jev_experiment.models import Ownership, ReviewStatus


def test_seed_dataset_contains_the_balanced_generated_baseline() -> None:
    cases = load_cases(Path("data/seeds/cases.csv"))
    summary = validate_case_collection(cases, load_reason_codes())

    assert summary.total >= 24
    assert all(summary.labels[label] >= 8 for label in Ownership)
    assert sum(summary.review_statuses.values()) == summary.total
    assert summary.paired_cases >= 12


def test_every_seed_case_is_approved() -> None:
    cases = load_cases(Path("data/seeds/cases.csv"))

    assert cases
    assert {case.review_status for case in cases} == {ReviewStatus.APPROVED}


def test_every_seed_pair_has_exactly_two_members_with_different_labels() -> None:
    cases = load_cases(Path("data/seeds/cases.csv"))
    pair_counts = Counter(case.pair_id for case in cases if case.pair_id)

    assert set(pair_counts.values()) == {2}
    for pair_id in pair_counts:
        labels = {case.gold_label for case in cases if case.pair_id == pair_id}
        assert len(labels) > 1, pair_id


def test_csv_round_trip_preserves_cases(tmp_path: Path) -> None:
    source_cases = load_cases(Path("data/seeds/cases.jsonl"))
    output = tmp_path / "cases.csv"

    from jev_experiment.dataset import write_cases_csv

    write_cases_csv(source_cases, output)

    assert load_cases(output) == source_cases
