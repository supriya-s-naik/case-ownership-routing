# Jev Ownership Classification Experiment

## 1. Objective

Build a reproducible LangGraph workflow that classifies synthetic Salesforce-style
support case histories into one of three ownership categories:

- `MINE`: the Platform Functionality L3 team owns the next concrete action.
- `NOT_MINE`: the Data and Methodology L3 team owns the next concrete action.
- `UNSURE`: the available evidence does not establish which team should act next.

Run the same cases through Jev, Claude Haiku, and Claude Sonnet, then compare
classification quality, consistency, latency, cost, and confidence behavior.

The goal is to evaluate whether Jev is competitive with general-purpose LLMs for
this bounded routing decision.

## 2. Fixed Decisions

### Ownership rule

Classify a case according to the team responsible for the **next concrete action**,
using the latest credible information in the case history.

### Team boundaries

`MINE` includes:

- Login, authentication, SSO, MFA, session, or account access failures.
- Entitlement, permission, license, or feature-access failures.
- Functional failures, error messages, exceptions, stuck workflows, and broken UI
  or API behavior.
- Debugging that may lead to a workaround or an Engineering bug report.

`NOT_MINE` includes:

- Incorrect or unexpected numbers displayed on dashboards or reports.
- Differences between pages, reports, exports, or external systems.
- Calculation, metric-definition, and methodology questions.
- Data reconciliation or requests to explain why a value was produced.

`UNSURE` includes:

- Vague reports such as "the dashboard is broken" without actionable evidence.
- Contradictory comments where the latest credible diagnosis is unclear.
- Multiple issues where the immediate next owner cannot be determined.
- Cases waiting for information required to distinguish functional failure from a
  data or methodology issue.

### Precedence examples

- "The dashboard shows the wrong number" is `NOT_MINE`.
- "The dashboard shows an error instead of a number" is `MINE`.
- "I cannot see the Risk tab" is `MINE` when evidence indicates access or
  entitlement.
- "How is Risk calculated?" is `NOT_MINE`.
- A later confirmed software exception overrides an earlier report of incorrect
  values and makes the next action `MINE`.
- A later confirmation that the software works as designed and the difference is
  methodological makes the next action `NOT_MINE`.

## 3. Scope and Non-Goals

### In scope

- Fully synthetic, fictional case histories.
- Free-form email and Salesforce-comment-style transcripts.
- Three-way ownership classification.
- A deterministic LangGraph benchmark workflow.
- Jev, Haiku, and Sonnet through OpenRouter.
- Offline evaluation against human-approved gold labels.
- Optional LangSmith tracing for synthetic data only.

### Out of scope for the first benchmark

- Real employer, client, Salesforce, or proprietary data.
- Drafting replies or resolving support cases.
- RAG, embeddings, or a vector database.
- Autonomous tool selection or a free-running agent loop.
- Using one model as the ground-truth judge for another model.
- Production deployment or automatic Salesforce routing.

## 4. Proposed Technology Stack

- Python 3.12
- `uv` for project and dependency management
- `langgraph` for workflow orchestration
- `langchain-core` for common runnable abstractions
- `langchain-openai` if an OpenRouter-compatible LangChain chat client is needed
- `typesafe-sdk` for Jev's typed decision contract
- `system-one-adapter[openai]` for running the same typed questions through Claude
  over OpenRouter where supported
- Pydantic for configuration, state, predictions, and dataset validation
- Pandas and scikit-learn for metrics and error analysis
- Matplotlib or Plotly for report charts
- Pytest for unit and integration tests
- Optional LangSmith for synthetic trace inspection and experiment comparison

All providers will use one `OPENROUTER_API_KEY`. Secrets will be read from the
environment or an ignored `.env` file and will never be committed.

## 5. Model Configuration

Use pinned model IDs so later runs remain comparable:

- Jev: `typesafe/jev-1.13`
- Haiku: `anthropic/claude-haiku-4.5`
- Sonnet: `anthropic/claude-sonnet-4.6`

The primary benchmark will give every model:

- The exact same case transcript.
- The exact same ownership policy and category criteria.
- The same three allowed category values.
- No access to the gold label, reason code, difficulty, tags, or other model output.

Claude models will use structured output, no extended reasoning, a small output
limit, and the lowest supported sampling variability. The exact request settings,
provider response metadata, served model ID, and raw output will be recorded.

The Jev Router model will not be used because it may route to non-Jev models.

## 6. Dataset Design

### Model-facing input

Each model receives a single unstructured transcript resembling a Salesforce case
history. It may contain:

- Subject and initial customer email.
- Customer follow-ups.
- L1 and L2 replies and internal notes.
- Forwarded or quoted messages.
- Timestamps, signatures, greetings, ticket references, and boilerplate.
- Typos, abbreviations, terse notes, and inconsistent terminology.
- Early diagnoses that are later corrected.
- Important evidence buried late in the thread.
- References to screenshots or attachments that are not available.

### Evaluation-only fields

The local dataset may use JSONL for storage, but only `case_text` is passed to the
classifier. A record will contain fields similar to:

```json
{
  "id": "CASE-0042",
  "case_text": "Subject: ...",
  "gold_label": "MINE",
  "reason_code": "ENTITLEMENT_MISSING",
  "scenario_family": "entitlement",
  "split": "test",
  "difficulty": "medium",
  "pair_id": null,
  "tags": ["misleading-number-language"]
}
```

### Dataset size and splits

Create 200 reviewed cases:

- 30 development cases for policy and prompt refinement.
- 150 frozen main-test cases, balanced at 50 per label.
- 20 robustness cases built as controlled pairs or transformations.

The test and robustness cases must remain untouched after the experiment is frozen.
The development cases are excluded from headline metrics.

### Scenario coverage

Cover at least these families:

- Login and authentication.
- Entitlement and access.
- Functional errors and failed workflows.
- Incorrect dashboard or report numbers.
- Page-to-page and system-to-system reconciliation.
- Methodology and metric-definition questions.
- Mixed, vague, contradictory, and insufficient-information cases.

### Synthetic-data safeguards

- Use a fictional company, products, people, IDs, and business domain.
- Do not copy real case language, customer details, identifiers, or proprietary
  methodology.
- Gold labels come from the scenario design and ownership policy, not from a model.
- A model may paraphrase an already labeled scenario, but it cannot select or alter
  the label.
- Manually review every frozen case for realism, label correctness, and leakage.
- Track `reason_code` and `scenario_family` to support error analysis.

## 7. LangGraph Architecture

Each graph invocation processes one case:

```text
                            +-> Jev node ------+
START -> validate case -----+-> Haiku node ----+-> join -> score -> analyze -> persist -> END
                            +-> Sonnet node ---+
```

The provider nodes are independent siblings. No model sees another model's output.
The join waits for all enabled providers before deterministic scoring.

Suggested graph state:

```python
class CaseState(TypedDict):
    case_id: str
    transcript: str
    expected_label: Ownership
    predictions: dict[str, Prediction]
    failures: dict[str, ProviderFailure]
    evaluation: CaseEvaluation | None
```

Core nodes:

1. `validate_case`: validate required fields without rewriting the transcript.
2. `classify_jev`: call the OpenRouter Decisions API using a typed `Choice`.
3. `classify_haiku`: call Haiku with the equivalent typed question contract.
4. `classify_sonnet`: call Sonnet with the equivalent typed question contract.
5. `score_case`: compare predictions with the hidden gold label in deterministic
   code.
6. `analyze_case`: mark disagreements, low-confidence results, schema failures, and
   robustness-pair inconsistencies.
7. `persist_result`: save raw and normalized results without exposing secrets.

Network nodes should use async clients and bounded retries for transient failures.
Latency must be measured inside each provider node so parallel graph execution does
not hide individual request time.

## 8. Repository Layout

```text
JevExperiment/
|-- PLAN.md
|-- README.md
|-- pyproject.toml
|-- uv.lock
|-- .env.example
|-- .gitignore
|-- config/
|   |-- ownership_policy.yaml
|   `-- experiment.yaml
|-- data/
|   |-- seeds/
|   |-- development.jsonl
|   |-- test.jsonl
|   `-- robustness.jsonl
|-- src/jev_experiment/
|   |-- models.py
|   |-- graph.py
|   |-- policy.py
|   |-- dataset.py
|   |-- providers/
|   |   |-- jev.py
|   |   `-- claude.py
|   |-- evaluation/
|   |   |-- metrics.py
|   |   `-- report.py
|   `-- cli.py
|-- tests/
|   |-- unit/
|   |-- integration/
|   `-- fixtures/
`-- results/
    `-- .gitkeep
```

Generated raw results should normally be ignored by Git. Small, redacted benchmark
summaries may be committed deliberately.

## 9. Evaluation Method

### Primary quality metrics

- Macro F1 across `MINE`, `NOT_MINE`, and `UNSURE`.
- Recall and precision for each label.
- Confusion matrix.
- Overall accuracy.

Macro F1 is the headline metric because all three categories matter even if a later
production-like distribution is imbalanced.

### Operational metrics

- Per-request cost and total experiment cost.
- Input and output token counts where available.
- p50 and p95 provider latency.
- API or schema failure rate.
- Retry count.
- Run-to-run label agreement across three repetitions.

### Confidence metrics

Where comparable probabilities are available:

- Brier score.
- Expected calibration error.
- Accuracy-versus-coverage curves.
- Accuracy after escalating low-confidence or `UNSURE` cases.

Claude's self-reported probabilities and Jev's native probabilities must be
identified as different mechanisms. Compare their empirical usefulness, not merely
their numeric values.

### Business-weighted view

Before freezing the test set, define and document error costs. A provisional rule is
that incorrectly returning `NOT_MINE` for an owned case is more costly than returning
`UNSURE`, because it may delay an issue the team should handle. Report both ordinary
metrics and the agreed business-weighted metric.

### Robustness analysis

Measure sensitivity to:

- Harmless paraphrasing.
- Reordered quoted history.
- Added signatures or irrelevant boilerplate.
- Misleading keywords.
- A single changed fact that should flip ownership.
- A later comment that supersedes an earlier diagnosis.

## 10. Experimental Controls

- Freeze policy text, model settings, dataset, and scoring code before the final run.
- Hash or version the policy and request configuration in every result record.
- Randomize provider execution order when performing sequential latency checks.
- Run every frozen case three times per model.
- Preserve raw responses and normalized predictions for auditability.
- Record both requested and actually served model IDs.
- Do not repair a model's semantic answer after inference.
- Treat malformed responses and exhausted retries as failures, not silently as
  `UNSURE`.
- Keep development-set results separate from final metrics.
- Use deterministic code, never another model, to compare predictions with labels.

## 11. Implementation Phases

### Phase 1: Scaffold and API spike

Deliverables:

- Initialize the `uv` project and dependencies.
- Add configuration loading and secret-safe `.env.example`.
- Define Pydantic dataset, prediction, and result models.
- Implement one three-case smoke test against Jev, Haiku, and Sonnet through
  OpenRouter.
- Confirm structured outputs, usage metadata, served model IDs, and cost reporting.

Exit criteria:

- All three pinned models return a valid category for the same sample transcript.
- No credentials or provider responses containing secrets are committed.

### Phase 2: Ownership policy and seed cases

Deliverables:

- Write the versioned ownership policy.
- Hand-design 20-30 representative seed scenarios.
- Include temporal updates, ambiguous cases, and minimal pairs.
- Review every seed against the "next concrete action" rule.

Exit criteria:

- Every seed has an agreed label and reason code.
- At least two people could apply the written policy without relying on unstated
  workplace knowledge; if only one reviewer is available, record that limitation.

### Phase 3: LangGraph workflow

Deliverables:

- Implement typed graph state and provider nodes.
- Add parallel fan-out, join, deterministic scoring, and persistence.
- Add retries, timeouts, and provider failure capture.
- Add an offline fixture provider so unit tests never require API access.

Exit criteria:

- Unit tests cover graph routing, join behavior, failures, and gold-label isolation.
- A development case produces a complete result record for every enabled provider.

### Phase 4: Synthetic dataset

Deliverables:

- Expand reviewed seeds into 200 cases.
- Create development, frozen test, and robustness splits.
- Validate balance, IDs, reason codes, pair relationships, and prohibited data.
- Run a manual leakage and realism review.

Exit criteria:

- Dataset validation passes with no duplicates or missing labels.
- Main test set contains exactly 50 examples of each label.
- No model-created label is accepted without human policy review.

### Phase 5: Development and freeze

Deliverables:

- Run only the development set while refining policy wording and provider adapters.
- Finalize error weights and confidence-handling rules.
- Freeze the policy, model settings, dataset, and scoring version.

Exit criteria:

- Test data has not been used to tune the prompt or policy.
- A manifest records hashes and model IDs for the frozen experiment.

### Phase 6: Final benchmark

Deliverables:

- Run Jev, Haiku, and Sonnet three times on all frozen cases.
- Store raw results and provider metadata.
- Rerun only documented transient failures; do not selectively rerun wrong answers.

Exit criteria:

- Every expected invocation is accounted for as success or explicit failure.
- Total spend remains within the configured experiment budget.

### Phase 7: Analysis and report

Deliverables:

- Generate metrics, confusion matrices, calibration plots, and cost/latency tables.
- Break down errors by scenario family, reason code, difficulty, and robustness tag.
- Inspect model disagreements and high-confidence errors manually.
- Write a concise report with conclusions and limitations.

Exit criteria:

- Results can be reproduced from saved predictions without making new API calls.
- The report distinguishes model quality from gateway latency and provider failures.

### Phase 8: Follow-on evaluation use case

After the ownership benchmark is complete, reuse the provider and evaluation
infrastructure for Jev-as-a-judge versus LLM-as-a-judge. Add this as a separate graph
and dataset so the routing benchmark remains frozen and interpretable.

## 12. Testing Strategy

- Unit-test policy loading, dataset validation, result normalization, and metrics.
- Test graph execution with fixture providers, including timeouts and malformed data.
- Test that provider payloads never contain `gold_label` or evaluation metadata.
- Add a small opt-in integration suite for live OpenRouter calls.
- Snapshot request shapes and normalized responses, not secrets or volatile IDs.
- Verify metric calculations against small hand-computed examples.

## 13. Cost Plan

Start with approximately $20 in OpenRouter credit.

For 200 cases, three repetitions, about 2,000 input tokens per case, and about 100
output tokens, the rough final-run estimate is:

- Jev 1.13: about $0.05.
- Haiku 4.5: about $1.50.
- Sonnet 4.6: about $4.50.
- Combined final benchmark: about $6 before retries and development calls.

These numbers are planning estimates. Capture actual provider-reported usage and cost
for every request. Check current model prices before the final run, and stop the run
when the configured budget is reached rather than allowing an unbounded retry loop.

## 14. Definition of Done

The first experiment is complete when:

- The ownership policy is explicit and versioned.
- The repository contains 200 synthetic, reviewed cases and no real company data.
- The LangGraph workflow runs all three providers from the same input contract.
- The frozen benchmark has three recorded repetitions per model.
- Results include quality, consistency, latency, cost, failure, and confidence
  analyses.
- Every headline result is reproducible from saved raw predictions.
- The final report states limitations, including synthetic-data bias and OpenRouter's
  contribution to observed latency.
