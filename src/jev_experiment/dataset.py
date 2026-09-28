"""JSONL dataset loading and validation."""

import csv
import json
from collections import Counter
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from jev_experiment.models import CaseRecord
from jev_experiment.models import Ownership, ReviewStatus


class ReasonCodeDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Ownership
    description: str


class DatasetSummary(BaseModel):
    total: int
    labels: dict[Ownership, int]
    review_statuses: dict[ReviewStatus, int]
    scenario_families: dict[str, int]
    paired_cases: int


CSV_FIELDS = (
    "id",
    "gold_label",
    "reason_code",
    "scenario_family",
    "difficulty",
    "review_status",
    "review_notes",
    "pair_id",
    "tags",
    "split",
    "case_text",
)


def _load_jsonl_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        records.append(value)
    return records


def _load_csv_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(CSV_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing CSV columns in {path}: {', '.join(sorted(missing))}")
        for row in reader:
            records.append(
                {
                    **row,
                    "pair_id": row["pair_id"].strip() or None,
                    "review_notes": row["review_notes"].strip() or None,
                    "tags": [tag.strip() for tag in row["tags"].split(";") if tag.strip()],
                }
            )
    return records


def load_cases(path: Path) -> list[CaseRecord]:
    """Load a JSONL or CSV dataset and reject duplicate case IDs."""

    cases: list[CaseRecord] = []
    if path.suffix.lower() == ".jsonl":
        records = _load_jsonl_records(path)
    elif path.suffix.lower() == ".csv":
        records = _load_csv_records(path)
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix}; use .jsonl or .csv")

    for record_number, record in enumerate(records, 1):
        try:
            cases.append(CaseRecord.model_validate(record))
        except ValueError as exc:
            raise ValueError(f"Invalid record at {path} record {record_number}: {exc}") from exc

    ids = [case.id for case in cases]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate case IDs: {', '.join(duplicates)}")
    return cases


def write_cases_csv(cases: list[CaseRecord], path: Path) -> None:
    """Write an Excel-friendly UTF-8 CSV for human review."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "id": case.id,
                    "gold_label": case.gold_label.value,
                    "reason_code": case.reason_code,
                    "scenario_family": case.scenario_family,
                    "difficulty": case.difficulty,
                    "review_status": case.review_status.value,
                    "review_notes": case.review_notes or "",
                    "pair_id": case.pair_id or "",
                    "tags": "; ".join(case.tags),
                    "split": case.split.value,
                    "case_text": case.case_text,
                }
            )


def load_reason_codes(
    path: Path = Path("config/reason_codes.yaml"),
) -> dict[str, ReasonCodeDefinition]:
    """Load the controlled reason-code catalog."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("reason_codes"), dict):
        raise ValueError(f"Expected a reason_codes mapping in {path}")
    return {
        code: ReasonCodeDefinition.model_validate(definition)
        for code, definition in payload["reason_codes"].items()
    }


def validate_case_collection(
    cases: list[CaseRecord],
    reason_codes: dict[str, ReasonCodeDefinition],
) -> DatasetSummary:
    """Validate cross-record invariants without imposing a maximum dataset size."""

    errors: list[str] = []
    normalized_texts: dict[str, str] = {}
    pair_counts = Counter(case.pair_id for case in cases if case.pair_id is not None)

    for case in cases:
        definition = reason_codes.get(case.reason_code)
        if definition is None:
            errors.append(f"{case.id}: unknown reason_code {case.reason_code}")
        elif definition.label != case.gold_label:
            errors.append(
                f"{case.id}: reason_code {case.reason_code} belongs to "
                f"{definition.label}, not {case.gold_label}"
            )

        normalized = " ".join(case.case_text.lower().split())
        if prior_id := normalized_texts.get(normalized):
            errors.append(f"{case.id}: duplicate transcript also used by {prior_id}")
        normalized_texts[normalized] = case.id

    for pair_id, count in sorted(pair_counts.items()):
        if count < 2:
            errors.append(f"{pair_id}: pair_id is used by only {count} case")

    if errors:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(errors))

    label_counts = Counter(case.gold_label for case in cases)
    review_counts = Counter(case.review_status for case in cases)
    family_counts = Counter(case.scenario_family for case in cases)
    return DatasetSummary(
        total=len(cases),
        labels={label: label_counts[label] for label in Ownership},
        review_statuses={status: review_counts[status] for status in ReviewStatus},
        scenario_families=dict(sorted(family_counts.items())),
        paired_cases=sum(count for count in pair_counts.values()),
    )
