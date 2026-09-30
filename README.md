# Jev Ownership Classification Experiment

This repository benchmarks Jev, Claude Haiku, and Claude Sonnet on a synthetic
three-way support-case ownership decision. See [PLAN.md](PLAN.md) for the full
experimental design.

## Local setup

1. Put your OpenRouter key in `.env`:

   ```dotenv
   OPENROUTER_API_KEY=your_key_here
   ```

2. Create the environment and run the offline tests:

   ```powershell
   uv sync
   uv run pytest
   ```

Live provider calls require an explicit smoke-test command. The normal test suite uses
offline fixture providers and will not spend API credit.

## Live smoke test

The smoke test runs the LangGraph workflow for three development cases against Jev,
Haiku, and Sonnet, making at least nine billable calls plus any configured retries. It
must be explicitly enabled:

```powershell
uv run jev-experiment smoke --live
```

Detailed JSON results and a human-readable Markdown statistics report are written
under `results/`, which is ignored by Git. The command never prints the API key.
The provider adapters retain OpenRouter-reported token and cost data plus generation
IDs when those fields are present in the response.

## Workflow

Each case runs through a typed LangGraph: validation, parallel independent provider
branches, deterministic scoring and analysis, then atomic JSON persistence. Provider
nodes receive only the ownership policy and case transcript; gold labels remain inside
the deterministic evaluation path.

## Budget-guarded benchmark

Preview an approved dataset without making API calls:

```powershell
uv run jev-experiment benchmark data/seeds/cases.csv
```

The preview validates that every case is approved, counts planned calls, and estimates
cost using the latest saved run with complete provider cost coverage. It adds a 25%
retry reserve. A live run requires both explicit billing authorization and a cost limit:

```powershell
uv run jev-experiment benchmark data/seeds/cases.csv --live --max-cost-usd 0.20
```

The command refuses to start when the guarded estimate exceeds the limit. During a
run it executes one case at a time, persists each completed result, and stops launching
new cases if the next reserved estimate would exceed the limit or cost visibility becomes
incomplete. Provider billing for failed attempts and already-running calls may not be
reported, so the limit is a guard rather than a guaranteed account-level hard cap.
An individual case that costs more than estimated can cross the limit before its three
parallel provider calls finish; the command records that case and stops before launching
another one.

## Visual benchmark dashboard

Launch the local dashboard:

```powershell
uv run jev-experiment ui
```

The dashboard previews the approved dataset and guarded cost estimate for free. A live
run starts only after you enter a sufficient cost limit, acknowledge billing, and press
the start button. Its model comparison cards show accuracy, Macro F1, mean latency,
provider cost, and per-provider completion progress. Completed runs also include
separate zero-based bar charts for accuracy, Macro F1, latency, and cost, plus per-label
precision/recall/F1 tables and confusion matrices. The execution-mode radio offers two
choices:
**Synchronized** waits for all three providers before starting the next case, while
**Throughput** gives each provider an independent sequential queue so a faster model can
advance immediately. It also shows side-by-side decisions,
confidence, probabilities, disagreements, and low-confidence flags inside a collapsed
diagnostics section. Budget controls are collapsed separately so the primary completed
view stays focused on model quality, latency, and cost. Completed cases are persisted by
the same benchmark engine used by the terminal command. Downloaded Markdown reports
include the same primary classification metrics.

Each provider request is stateless and contains only the ownership policy and the current
case transcript. Previous runs and predictions are not sent to the models. Because the
request does not pin a temperature or random seed, repeated live runs can still differ on
borderline cases.

## Seed cases

Approved seed cases live in `data/seeds/cases.csv`. You can add new draft cases for
review; follow `data/seeds/README.md` and validate changes with:

```powershell
uv run jev-experiment validate-data data/seeds/cases.csv
```
