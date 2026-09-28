# Seed-case authoring guide

Use `cases.csv` as the editable human-review source. The current records are synthetic
and approved, and adding more draft cases is encouraged. `cases.jsonl` is the original
generated snapshot; you do not need to edit it.

Each CSV row must contain:

- A unique ID such as `SEED-0025`.
- A fully fictional `case_text` transcript.
- One `gold_label`: `MINE`, `NOT_MINE`, or `UNSURE`.
- A `reason_code` listed in `config/reason_codes.yaml` and belonging to that label.
- A short `scenario_family`, `difficulty`, and optional tags.
- `split: "development"` while the policy is still being refined.
- `review_status: "draft"` until a human approves the case and label.
- An optional `pair_id`; every pair ID must occur on at least two cases.
- Optional `review_notes` for your comments or requested revisions.
- Semicolon-separated values in `tags`, such as `sso; reproduced`.

Do not copy real customer language, identifiers, people, products, companies, or
methodology. A case can be edited freely while it is a draft. After approval, make
substantive changes by creating a new draft rather than silently changing the
approved gold example.

Validate after editing:

```powershell
uv run jev-experiment validate-data data/seeds/cases.csv
```

## Review record

The 26 seed labels were approved by one human reviewer. No independent second-reviewer
agreement measurement is available for this seed set; treat that as a limitation when
interpreting benchmark accuracy.
