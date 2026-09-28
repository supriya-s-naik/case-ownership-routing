from pathlib import Path

from jev_experiment.dataset import load_cases
from jev_experiment.models import Ownership


def test_development_smoke_cases_cover_all_labels() -> None:
    cases = load_cases(Path("data/development.jsonl"))

    assert len(cases) == 3
    assert {case.gold_label for case in cases} == set(Ownership)
