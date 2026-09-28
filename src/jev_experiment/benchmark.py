"""Budget-aware planning and execution for approved benchmark datasets."""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from jev_experiment.config import load_settings
from jev_experiment.dataset import load_cases, load_reason_codes, validate_case_collection
from jev_experiment.evaluation import render_smoke_report, reported_cost
from jev_experiment.experiment import load_experiment_config
from jev_experiment.graph import build_case_graph, classify_with_retry, run_case_graph
from jev_experiment.models import (
    CaseEvaluation,
    CaseRecord,
    CaseResult,
    Prediction,
    ProviderFailure,
    ReviewStatus,
)
from jev_experiment.persistence import JsonResultStore
from jev_experiment.providers.base import ClassificationRequest, ClassifierProvider
from jev_experiment.providers.contract import load_policy_text
from jev_experiment.providers.live import build_live_providers


DEFAULT_RETRY_RESERVE = 0.25
SYNCHRONIZED_MODE = "synchronized"
THROUGHPUT_MODE = "throughput"
EXECUTION_MODES = (SYNCHRONIZED_MODE, THROUGHPUT_MODE)


@dataclass(frozen=True)
class BenchmarkPlan:
    dataset_path: Path
    cases: tuple[CaseRecord, ...]
    models: dict[str, str]
    repetitions: int
    planned_calls: int
    estimate_source: Path
    estimated_cost_by_provider: dict[str, float]
    estimated_cost_per_case: float
    estimated_cost_usd: float
    retry_reserve: float
    guarded_estimate_usd: float


@dataclass(frozen=True)
class BenchmarkExecution:
    results: tuple[CaseResult, ...]
    output_path: Path
    report_path: Path
    actual_reported_cost_usd: float
    stopped_reason: str | None


@dataclass(frozen=True)
class BenchmarkProgress:
    stage: str
    case_id: str
    repetition: int
    completed_case_runs: int
    total_case_runs: int
    actual_reported_cost_usd: float
    result: CaseResult | None = None
    provider: str | None = None
    provider_completed_calls: int | None = None
    prediction: Prediction | None = None
    failure: ProviderFailure | None = None
    provider_elapsed_seconds: float | None = None


def _load_saved_results(path: Path) -> list[CaseResult]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON result list in {path}")
    return [CaseResult.model_validate(item) for item in payload]


def _cost_estimates_from_results(
    path: Path,
    provider_names: tuple[str, ...],
) -> dict[str, float]:
    results = _load_saved_results(path)
    estimates: dict[str, float] = {}
    for provider in provider_names:
        predictions = [
            result.predictions[provider]
            for result in results
            if provider in result.predictions
        ]
        if not predictions:
            raise ValueError(f"No successful {provider} calls in cost source {path}")
        costs = [reported_cost(prediction) for prediction in predictions]
        if any(cost is None for cost in costs):
            raise ValueError(f"Incomplete {provider} cost coverage in {path}")
        estimates[provider] = sum(float(cost) for cost in costs if cost is not None) / len(
            costs
        )
    return estimates


def _latest_usable_cost_source(
    results_dir: Path,
    provider_names: tuple[str, ...],
) -> tuple[Path, dict[str, float]]:
    errors: list[str] = []
    candidates = sorted(
        results_dir.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        try:
            return candidate, _cost_estimates_from_results(candidate, provider_names)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{candidate.name}: {exc}")
    detail = f" Checked: {'; '.join(errors)}" if errors else ""
    raise ValueError(
        "No saved result file has complete provider cost coverage. "
        "Run the live smoke test or pass --estimate-from." + detail
    )


def prepare_benchmark_plan(
    dataset_path: Path,
    *,
    repetitions: int = 1,
    estimate_from: Path | None = None,
    results_dir: Path = Path("results"),
    retry_reserve: float = DEFAULT_RETRY_RESERVE,
) -> BenchmarkPlan:
    """Validate an approved dataset and calculate a non-billable execution plan."""

    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    if retry_reserve < 0:
        raise ValueError("retry_reserve cannot be negative")

    cases = load_cases(dataset_path)
    validate_case_collection(cases, load_reason_codes())
    if not cases:
        raise ValueError("Benchmark dataset is empty")
    draft_ids = [case.id for case in cases if case.review_status is not ReviewStatus.APPROVED]
    if draft_ids:
        raise ValueError(
            "Benchmark requires approved cases; drafts: " + ", ".join(draft_ids)
        )

    config = load_experiment_config()
    raw_models = config.get("models")
    if not isinstance(raw_models, dict) or not all(
        isinstance(name, str) and isinstance(model, str)
        for name, model in raw_models.items()
    ):
        raise ValueError("models configuration must map provider names to model IDs")
    models = dict(raw_models)
    provider_names = tuple(models)

    if estimate_from is None:
        estimate_source, estimates = _latest_usable_cost_source(
            results_dir,
            provider_names,
        )
    else:
        estimate_source = estimate_from
        estimates = _cost_estimates_from_results(estimate_source, provider_names)

    estimated_cost_per_case = sum(estimates.values())
    estimated_cost_usd = estimated_cost_per_case * len(cases) * repetitions
    planned_calls = len(cases) * len(provider_names) * repetitions
    return BenchmarkPlan(
        dataset_path=dataset_path,
        cases=tuple(cases),
        models=models,
        repetitions=repetitions,
        planned_calls=planned_calls,
        estimate_source=estimate_source,
        estimated_cost_by_provider=estimates,
        estimated_cost_per_case=estimated_cost_per_case,
        estimated_cost_usd=estimated_cost_usd,
        retry_reserve=retry_reserve,
        guarded_estimate_usd=estimated_cost_usd * (1 + retry_reserve),
    )


def format_benchmark_plan(plan: BenchmarkPlan) -> str:
    provider_costs = ", ".join(
        f"{provider}=${cost:.6f}/case"
        for provider, cost in plan.estimated_cost_by_provider.items()
    )
    return "\n".join(
        [
            "Benchmark preview (no API calls made)",
            f"Dataset: {plan.dataset_path}",
            f"Approved cases: {len(plan.cases)}",
            f"Providers: {', '.join(plan.models)}",
            f"Repetitions: {plan.repetitions}",
            f"Planned calls: {plan.planned_calls}",
            f"Cost estimate source: {plan.estimate_source}",
            f"Observed average costs: {provider_costs}",
            f"Base estimated cost: ${plan.estimated_cost_usd:.6f}",
            f"Retry reserve: {plan.retry_reserve:.0%}",
            f"Budget-guarded estimate: ${plan.guarded_estimate_usd:.6f}",
        ]
    )


def _benchmark_report(
    results: list[CaseResult],
    plan: BenchmarkPlan,
    *,
    maximum_cost_usd: float,
    actual_reported_cost_usd: float,
    stopped_reason: str | None,
    execution_mode: str,
) -> str:
    provider_report = render_smoke_report(results).replace(
        "# Provider Smoke-Test Report",
        "# Case Ownership Routing Report",
        1,
    )
    status = "stopped early" if stopped_reason else "complete"
    budget_lines = [
        "",
        "## Execution and budget",
        "",
        f"- Status: {status}",
        f"- Execution mode: {execution_mode}",
        f"- Dataset: `{plan.dataset_path}`",
        f"- Planned provider calls: {plan.planned_calls}",
        f"- Completed case runs: {len(results)} of {len(plan.cases) * plan.repetitions}",
        f"- Base estimated cost: ${plan.estimated_cost_usd:.6f}",
        f"- Budget-guarded estimate: ${plan.guarded_estimate_usd:.6f}",
        f"- User cost limit: ${maximum_cost_usd:.6f}",
        f"- Actual provider-reported cost: ${actual_reported_cost_usd:.6f}",
    ]
    if stopped_reason:
        budget_lines.append(f"- Stop reason: {stopped_reason}")
    lines = provider_report.splitlines()
    return "\n".join([lines[0], *budget_lines, "", *lines[1:]])


def _case_result_from_outcomes(
    case: CaseRecord,
    repetition: int,
    predictions: dict[str, Prediction],
    failures: dict[str, ProviderFailure],
    *,
    low_confidence_threshold: float,
) -> CaseResult:
    labels = {prediction.label for prediction in predictions.values()}
    evaluation = CaseEvaluation(
        correct_by_provider={
            provider: prediction.label == case.gold_label
            for provider, prediction in sorted(predictions.items())
        },
        disagreement=len(labels) > 1,
        low_confidence_providers=sorted(
            provider
            for provider, prediction in predictions.items()
            if prediction.confidence is not None
            and prediction.confidence < low_confidence_threshold
        ),
        failure_providers=sorted(failures),
    )
    return CaseResult(
        case_id=case.id,
        repetition=repetition,
        expected_label=case.gold_label,
        predictions=dict(sorted(predictions.items())),
        failures=dict(sorted(failures.items())),
        evaluation=evaluation,
        created_at=datetime.now(UTC),
    )


async def _execute_throughput_benchmark(
    plan: BenchmarkPlan,
    providers: tuple[ClassifierProvider, ...],
    *,
    policy_text: str,
    maximum_cost_usd: float,
    output_path: Path,
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
    low_confidence_threshold: float,
    progress_callback: Callable[[BenchmarkProgress], None] | None,
    should_stop: Callable[[], bool] | None,
) -> BenchmarkExecution:
    """Run one sequential queue per provider and join outcomes by case."""

    store = JsonResultStore(output_path)
    work_items = tuple(
        (repetition, case)
        for repetition in range(1, plan.repetitions + 1)
        for case in plan.cases
    )
    total_case_runs = len(work_items)
    provider_names = {provider.name for provider in providers}
    provider_completed_calls = {provider.name: 0 for provider in providers}
    provider_started_at = {provider.name: perf_counter() for provider in providers}
    predictions_by_key: dict[tuple[str, int], dict[str, Prediction]] = {}
    failures_by_key: dict[tuple[str, int], dict[str, ProviderFailure]] = {}
    case_by_key = {(case.id, repetition): case for repetition, case in work_items}
    work_order = {
        (case.id, repetition): index
        for index, (repetition, case) in enumerate(work_items)
    }
    started_keys: set[tuple[str, int]] = set()
    persisted_keys: set[tuple[str, int]] = set()
    results: list[CaseResult] = []
    state_lock = asyncio.Lock()
    actual_cost = 0.0
    inflight_reserve = 0.0
    stopped_reason: str | None = None

    async def provider_worker(provider: ClassifierProvider) -> None:
        nonlocal actual_cost, inflight_reserve, stopped_reason
        reserved_call_cost = plan.estimated_cost_by_provider[provider.name] * (
            1 + plan.retry_reserve
        )
        for repetition, case in work_items:
            key = (case.id, repetition)
            async with state_lock:
                if should_stop is not None and should_stop():
                    if stopped_reason is None:
                        stopped_reason = "User requested stop after active provider calls"
                    break
                if stopped_reason is not None:
                    break
                if actual_cost + inflight_reserve + reserved_call_cost > maximum_cost_usd:
                    stopped_reason = (
                        "Launching the next provider call would exceed the cost limit "
                        "using the reserved per-provider estimate"
                    )
                    break
                inflight_reserve += reserved_call_cost
                started_keys.add(key)
                if progress_callback is not None:
                    progress_callback(
                        BenchmarkProgress(
                            stage="provider_running",
                            case_id=case.id,
                            repetition=repetition,
                            completed_case_runs=len(results),
                            total_case_runs=total_case_runs,
                            actual_reported_cost_usd=actual_cost,
                            provider=provider.name,
                            provider_completed_calls=provider_completed_calls[
                                provider.name
                            ],
                            provider_elapsed_seconds=(
                                perf_counter() - provider_started_at[provider.name]
                            ),
                        )
                    )

            request = ClassificationRequest(
                policy_text=policy_text,
                case_transcript=case.case_text,
            )
            outcome = await classify_with_retry(
                provider,
                request,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )

            async with state_lock:
                inflight_reserve -= reserved_call_cost
                provider_completed_calls[provider.name] += 1
                prediction = outcome if isinstance(outcome, Prediction) else None
                failure = outcome if isinstance(outcome, ProviderFailure) else None
                if prediction is not None:
                    predictions_by_key.setdefault(key, {})[provider.name] = prediction
                    cost = reported_cost(prediction)
                    if cost is None:
                        if stopped_reason is None:
                            stopped_reason = (
                                "Budget visibility became incomplete (missing reported "
                                f"cost: {provider.name})"
                            )
                    else:
                        actual_cost += cost
                else:
                    assert failure is not None
                    failures_by_key.setdefault(key, {})[provider.name] = failure
                    if stopped_reason is None:
                        stopped_reason = (
                            "Budget visibility became incomplete (provider failure: "
                            f"{provider.name})"
                        )

                elapsed = perf_counter() - provider_started_at[provider.name]
                if progress_callback is not None:
                    progress_callback(
                        BenchmarkProgress(
                            stage="provider_completed",
                            case_id=case.id,
                            repetition=repetition,
                            completed_case_runs=len(results),
                            total_case_runs=total_case_runs,
                            actual_reported_cost_usd=actual_cost,
                            provider=provider.name,
                            provider_completed_calls=provider_completed_calls[
                                provider.name
                            ],
                            prediction=prediction,
                            failure=failure,
                            provider_elapsed_seconds=elapsed,
                        )
                    )

                completed_providers = set(predictions_by_key.get(key, {})) | set(
                    failures_by_key.get(key, {})
                )
                if completed_providers == provider_names:
                    result = _case_result_from_outcomes(
                        case,
                        repetition,
                        predictions_by_key.get(key, {}),
                        failures_by_key.get(key, {}),
                        low_confidence_threshold=low_confidence_threshold,
                    )
                    await store.save(result)
                    persisted_keys.add(key)
                    results.append(result)
                    if progress_callback is not None:
                        progress_callback(
                            BenchmarkProgress(
                                stage="completed",
                                case_id=case.id,
                                repetition=repetition,
                                completed_case_runs=len(results),
                                total_case_runs=total_case_runs,
                                actual_reported_cost_usd=actual_cost,
                                result=result,
                            )
                        )
                if actual_cost > maximum_cost_usd and stopped_reason is None:
                    stopped_reason = (
                        f"Actual provider-reported cost ${actual_cost:.6f} exceeded "
                        f"the ${maximum_cost_usd:.6f} limit within active calls"
                    )

    await asyncio.gather(*(provider_worker(provider) for provider in providers))

    for key in sorted(started_keys - persisted_keys, key=work_order.__getitem__):
        case = case_by_key[key]
        result = _case_result_from_outcomes(
            case,
            key[1],
            predictions_by_key.get(key, {}),
            failures_by_key.get(key, {}),
            low_confidence_threshold=low_confidence_threshold,
        )
        await store.save(result)
        results.append(result)
        if progress_callback is not None:
            progress_callback(
                BenchmarkProgress(
                    stage="partial",
                    case_id=key[0],
                    repetition=key[1],
                    completed_case_runs=len(results),
                    total_case_runs=total_case_runs,
                    actual_reported_cost_usd=actual_cost,
                    result=result,
                )
            )

    results.sort(key=lambda result: work_order[(result.case_id, result.repetition)])
    report_path = output_path.with_name(f"{output_path.stem}-summary.md")
    report_path.write_text(
        _benchmark_report(
            results,
            plan,
            maximum_cost_usd=maximum_cost_usd,
            actual_reported_cost_usd=actual_cost,
            stopped_reason=stopped_reason,
            execution_mode=THROUGHPUT_MODE,
        ),
        encoding="utf-8",
    )
    return BenchmarkExecution(
        results=tuple(results),
        output_path=output_path,
        report_path=report_path,
        actual_reported_cost_usd=actual_cost,
        stopped_reason=stopped_reason,
    )


async def execute_benchmark(
    plan: BenchmarkPlan,
    providers: tuple[ClassifierProvider, ...],
    *,
    policy_text: str,
    maximum_cost_usd: float,
    output_path: Path,
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
    low_confidence_threshold: float,
    progress_callback: Callable[[BenchmarkProgress], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    execution_mode: str = SYNCHRONIZED_MODE,
) -> BenchmarkExecution:
    """Run a prepared plan in synchronized or independent-throughput mode."""

    if maximum_cost_usd <= 0:
        raise ValueError("maximum_cost_usd must be positive")
    if execution_mode not in EXECUTION_MODES:
        raise ValueError(
            f"execution_mode must be one of: {', '.join(EXECUTION_MODES)}"
        )
    provider_names = [provider.name for provider in providers]
    if set(provider_names) != set(plan.models):
        raise ValueError(
            "Enabled providers must match the plan: expected "
            f"{', '.join(plan.models)}, got {', '.join(provider_names)}"
        )
    if plan.guarded_estimate_usd > maximum_cost_usd:
        raise ValueError(
            f"Budget-guarded estimate ${plan.guarded_estimate_usd:.6f} exceeds "
            f"the ${maximum_cost_usd:.6f} limit; no API calls were made"
        )

    if execution_mode == THROUGHPUT_MODE:
        return await _execute_throughput_benchmark(
            plan,
            providers,
            policy_text=policy_text,
            maximum_cost_usd=maximum_cost_usd,
            output_path=output_path,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            low_confidence_threshold=low_confidence_threshold,
            progress_callback=progress_callback,
            should_stop=should_stop,
        )

    store = JsonResultStore(output_path)
    provider_completed_calls = {provider.name: 0 for provider in providers}
    inflight_reported_cost = 0.0

    def provider_completed(
        case_id: str,
        repetition: int,
        provider: str,
        outcome: Prediction | ProviderFailure,
    ) -> None:
        nonlocal inflight_reported_cost
        provider_completed_calls[provider] += 1
        prediction = outcome if isinstance(outcome, Prediction) else None
        failure = outcome if isinstance(outcome, ProviderFailure) else None
        if prediction is not None:
            cost = reported_cost(prediction)
            if cost is not None:
                inflight_reported_cost += cost
        if progress_callback is not None:
            progress_callback(
                BenchmarkProgress(
                    stage="provider_completed",
                    case_id=case_id,
                    repetition=repetition,
                    completed_case_runs=len(results),
                    total_case_runs=total_case_runs,
                    actual_reported_cost_usd=actual_cost + inflight_reported_cost,
                    provider=provider,
                    provider_completed_calls=provider_completed_calls[provider],
                    prediction=prediction,
                    failure=failure,
                )
            )

    graph = build_case_graph(
        providers,
        store=store,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        low_confidence_threshold=low_confidence_threshold,
        provider_progress_callback=provider_completed,
    )
    results: list[CaseResult] = []
    actual_cost = 0.0
    stopped_reason: str | None = None
    guarded_next_case = plan.estimated_cost_per_case * (1 + plan.retry_reserve)
    total_case_runs = len(plan.cases) * plan.repetitions

    for repetition in range(1, plan.repetitions + 1):
        for case in plan.cases:
            if should_stop is not None and should_stop():
                stopped_reason = "User requested stop after the completed case"
                break
            if actual_cost + guarded_next_case > maximum_cost_usd:
                stopped_reason = (
                    "Launching the next case would exceed the cost limit using the "
                    "reserved per-case estimate"
                )
                break
            if progress_callback is not None:
                progress_callback(
                    BenchmarkProgress(
                        stage="running",
                        case_id=case.id,
                        repetition=repetition,
                        completed_case_runs=len(results),
                        total_case_runs=total_case_runs,
                        actual_reported_cost_usd=actual_cost,
                    )
                )
            inflight_reported_cost = 0.0
            result = await run_case_graph(
                graph,
                case,
                policy_text=policy_text,
                repetition=repetition,
            )
            results.append(result)
            costs = [reported_cost(prediction) for prediction in result.predictions.values()]
            actual_cost += sum(float(cost) for cost in costs if cost is not None)
            if progress_callback is not None:
                progress_callback(
                    BenchmarkProgress(
                        stage="completed",
                        case_id=case.id,
                        repetition=repetition,
                        completed_case_runs=len(results),
                        total_case_runs=total_case_runs,
                        actual_reported_cost_usd=actual_cost,
                        result=result,
                    )
                )
            missing_costs = [
                provider
                for provider, prediction in result.predictions.items()
                if reported_cost(prediction) is None
            ]
            if result.failures or missing_costs:
                reasons: list[str] = []
                if result.failures:
                    reasons.append(
                        "provider failures: " + ", ".join(sorted(result.failures))
                    )
                if missing_costs:
                    reasons.append(
                        "missing reported costs: " + ", ".join(sorted(missing_costs))
                    )
                stopped_reason = "Budget visibility became incomplete (" + "; ".join(reasons) + ")"
                break
            if actual_cost > maximum_cost_usd:
                stopped_reason = (
                    f"Actual provider-reported cost ${actual_cost:.6f} exceeded "
                    f"the ${maximum_cost_usd:.6f} limit within the completed case"
                )
                break
        if stopped_reason:
            break

    report_path = output_path.with_name(f"{output_path.stem}-summary.md")
    report_path.write_text(
        _benchmark_report(
            results,
            plan,
            maximum_cost_usd=maximum_cost_usd,
            actual_reported_cost_usd=actual_cost,
            stopped_reason=stopped_reason,
            execution_mode=SYNCHRONIZED_MODE,
        ),
        encoding="utf-8",
    )
    return BenchmarkExecution(
        results=tuple(results),
        output_path=output_path,
        report_path=report_path,
        actual_reported_cost_usd=actual_cost,
        stopped_reason=stopped_reason,
    )


async def run_live_benchmark(
    plan: BenchmarkPlan,
    *,
    maximum_cost_usd: float,
    progress_callback: Callable[[BenchmarkProgress], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    execution_mode: str = SYNCHRONIZED_MODE,
) -> BenchmarkExecution:
    """Execute a prepared plan using the configured live OpenRouter providers."""

    settings = load_settings()
    config = load_experiment_config()
    request_config = config.get("request")
    if not isinstance(request_config, dict):
        raise ValueError("request configuration must be a mapping")
    timeout_seconds = float(request_config["timeout_seconds"])
    providers = build_live_providers(
        api_key=settings.openrouter_api_key.get_secret_value(),
        models=plan.models,
        timeout_seconds=timeout_seconds,
    )
    mode_segment = "throughput-" if execution_mode == THROUGHPUT_MODE else ""
    output_path = (
        Path("results")
        / f"benchmark-{mode_segment}{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    )
    return await execute_benchmark(
        plan,
        providers,
        policy_text=load_policy_text(),
        maximum_cost_usd=maximum_cost_usd,
        output_path=output_path,
        timeout_seconds=timeout_seconds,
        max_retries=int(request_config["max_retries"]),
        retry_backoff_seconds=float(request_config.get("retry_backoff_seconds", 0.25)),
        low_confidence_threshold=float(
            request_config.get("low_confidence_threshold", 0.5)
        ),
        progress_callback=progress_callback,
        should_stop=should_stop,
        execution_mode=execution_mode,
    )


def run_live_benchmark_sync(
    plan: BenchmarkPlan,
    *,
    maximum_cost_usd: float,
    progress_callback: Callable[[BenchmarkProgress], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    execution_mode: str = SYNCHRONIZED_MODE,
) -> BenchmarkExecution:
    return asyncio.run(
        run_live_benchmark(
            plan,
            maximum_cost_usd=maximum_cost_usd,
            progress_callback=progress_callback,
            should_stop=should_stop,
            execution_mode=execution_mode,
        )
    )
