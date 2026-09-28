"""Experiment configuration and live smoke-run orchestration."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from jev_experiment.config import load_settings
from jev_experiment.dataset import load_cases
from jev_experiment.evaluation import render_smoke_report
from jev_experiment.graph import build_case_graph, run_case_graph
from jev_experiment.models import CaseResult
from jev_experiment.persistence import JsonResultStore
from jev_experiment.providers.contract import load_policy_text
from jev_experiment.providers.live import build_live_providers


def load_experiment_config(path: Path = Path("config/experiment.yaml")) -> dict[str, object]:
    """Load the checked-in non-secret experiment configuration."""

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


async def run_live_smoke() -> tuple[list[CaseResult], Path, Path]:
    """Run the three development cases and persist their results."""

    settings = load_settings()
    config = load_experiment_config()
    request_config = config["request"]
    if not isinstance(request_config, dict):
        raise ValueError("request configuration must be a mapping")
    models = config["models"]
    if not isinstance(models, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in models.items()):
        raise ValueError("models configuration must map provider names to model IDs")

    cases = load_cases(Path("data/development.jsonl"))
    policy_text = load_policy_text()
    api_key = settings.openrouter_api_key.get_secret_value()
    output = Path("results") / f"smoke-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    store = JsonResultStore(output)
    providers = build_live_providers(
        api_key=api_key,
        models=models,
        timeout_seconds=float(request_config["timeout_seconds"]),
    )
    graph = build_case_graph(
        providers,
        store=store,
        timeout_seconds=float(request_config["timeout_seconds"]),
        max_retries=int(request_config["max_retries"]),
        retry_backoff_seconds=float(request_config.get("retry_backoff_seconds", 0.25)),
        low_confidence_threshold=float(
            request_config.get("low_confidence_threshold", 0.5)
        ),
    )
    results: list[CaseResult] = []
    for case in cases:
        results.append(
            await run_case_graph(
                graph,
                case,
                policy_text=policy_text,
            )
        )

    report_output = output.with_name(f"{output.stem}-summary.md")
    report_output.write_text(render_smoke_report(results), encoding="utf-8")
    return results, output, report_output


def run_live_smoke_sync() -> tuple[list[CaseResult], Path, Path]:
    return asyncio.run(run_live_smoke())


def render_saved_smoke_report(results_path: Path) -> Path:
    """Regenerate a Markdown report from saved results without API calls."""

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    results = [CaseResult.model_validate(item) for item in payload]
    report_path = results_path.with_name(f"{results_path.stem}-summary.md")
    report_path.write_text(render_smoke_report(results), encoding="utf-8")
    return report_path
