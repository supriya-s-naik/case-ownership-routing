"""Local Streamlit dashboard for previewing and running the benchmark."""

import json
import math
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

import altair as alt
import streamlit as st

from jev_experiment.benchmark import (
    BenchmarkExecution,
    BenchmarkPlan,
    BenchmarkProgress,
    SYNCHRONIZED_MODE,
    THROUGHPUT_MODE,
    prepare_benchmark_plan,
    run_live_benchmark_sync,
)
from jev_experiment.evaluation import (
    LABELS,
    classification_metrics,
    render_smoke_report,
    reported_cost,
)
from jev_experiment.models import CaseResult, Ownership, Prediction


@dataclass
class DashboardJob:
    plan: BenchmarkPlan
    maximum_cost_usd: float | None
    execution_mode: str = SYNCHRONIZED_MODE
    events: queue.Queue[BenchmarkProgress] = field(default_factory=queue.Queue)
    stop_event: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)
    results: dict[tuple[str, int], CaseResult] = field(default_factory=dict)
    provider_completed_calls: dict[str, int] = field(default_factory=dict)
    provider_cost_usd: dict[str, float] = field(default_factory=dict)
    provider_failures: dict[str, int] = field(default_factory=dict)
    provider_predictions: dict[tuple[str, int, str], Prediction] = field(
        default_factory=dict
    )
    provider_elapsed_seconds: dict[str, float] = field(default_factory=dict)
    provider_current_case: dict[str, tuple[str, int]] = field(default_factory=dict)
    current_case: tuple[str, int] | None = None
    actual_cost_usd: float = 0.0
    execution: BenchmarkExecution | None = None
    error: str | None = None
    thread: threading.Thread | None = None


def _run_job(job: DashboardJob) -> None:
    try:
        job.execution = run_live_benchmark_sync(
            job.plan,
            maximum_cost_usd=job.maximum_cost_usd,
            progress_callback=job.events.put,
            should_stop=job.stop_event.is_set,
            execution_mode=job.execution_mode,
        )
    except Exception as exc:  # displayed without exposing environment values
        job.error = f"{type(exc).__name__}: {exc}"
    finally:
        job.done_event.set()


def _start_job(
    plan: BenchmarkPlan,
    maximum_cost_usd: float,
    execution_mode: str,
) -> DashboardJob:
    job = DashboardJob(
        plan=plan,
        maximum_cost_usd=maximum_cost_usd,
        execution_mode=execution_mode,
        provider_completed_calls={provider: 0 for provider in plan.models},
        provider_cost_usd={provider: 0.0 for provider in plan.models},
        provider_failures={provider: 0 for provider in plan.models},
    )
    job.thread = threading.Thread(
        target=_run_job,
        args=(job,),
        name="jev-benchmark-dashboard",
        daemon=True,
    )
    job.thread.start()
    return job


def _latest_saved_benchmark(results_dir: Path = Path("results")) -> Path | None:
    candidates = sorted(
        results_dir.glob("benchmark-*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_saved_job(plan: BenchmarkPlan, path: Path) -> DashboardJob:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON result list in {path}")
    loaded_results = [CaseResult.model_validate(item) for item in payload]
    allowed_keys = {
        (case.id, repetition)
        for repetition in range(1, plan.repetitions + 1)
        for case in plan.cases
    }
    result_keys = [(result.case_id, result.repetition) for result in loaded_results]
    invalid_keys = [key for key in result_keys if key not in allowed_keys]
    if invalid_keys:
        raise ValueError(
            f"Saved run does not match the current dataset: {invalid_keys[0]}"
        )
    if len(result_keys) != len(set(result_keys)):
        raise ValueError(f"Saved run contains duplicate case results: {path}")

    job = DashboardJob(
        plan=plan,
        maximum_cost_usd=None,
        execution_mode=(
            THROUGHPUT_MODE
            if path.name.startswith("benchmark-throughput-")
            else SYNCHRONIZED_MODE
        ),
        provider_completed_calls={provider: 0 for provider in plan.models},
        provider_cost_usd={provider: 0.0 for provider in plan.models},
        provider_failures={provider: 0 for provider in plan.models},
    )
    for result in loaded_results:
        key = (result.case_id, result.repetition)
        job.results[key] = result
        job.current_case = key
        for provider, prediction in result.predictions.items():
            if provider not in job.provider_completed_calls:
                raise ValueError(f"Saved run contains unknown provider: {provider}")
            job.provider_completed_calls[provider] += 1
            job.provider_predictions[(result.case_id, result.repetition, provider)] = (
                prediction
            )
            job.provider_elapsed_seconds[provider] = (
                job.provider_elapsed_seconds.get(provider, 0.0)
                + prediction.latency_ms / 1000
            )
            cost = reported_cost(prediction)
            if cost is not None:
                job.provider_cost_usd[provider] += cost
                job.actual_cost_usd += cost
        for provider in result.failures:
            if provider not in job.provider_completed_calls:
                raise ValueError(f"Saved run contains unknown provider: {provider}")
            job.provider_completed_calls[provider] += 1
            job.provider_failures[provider] += 1
    report_path = path.with_name(f"{path.stem}-summary.md")
    job.execution = BenchmarkExecution(
        results=tuple(loaded_results),
        output_path=path,
        report_path=report_path,
        actual_reported_cost_usd=job.actual_cost_usd,
        stopped_reason=(
            None
            if len(loaded_results) == len(allowed_keys)
            else "Loaded saved run is partial"
        ),
    )
    job.done_event.set()
    return job


def _downloadable_report(job: DashboardJob) -> str:
    """Render current metrics while preserving saved execution metadata when present."""

    if job.execution is None:
        return ""
    refreshed = render_smoke_report(list(job.execution.results)).replace(
        "# Provider Smoke-Test Report",
        "# Case Ownership Routing Report",
        1,
    )
    report_path = job.execution.report_path
    if not report_path.exists():
        return refreshed
    existing = report_path.read_text(encoding="utf-8")
    marker = "\nCases:"
    existing_body = existing.find(marker)
    refreshed_body = refreshed.find(marker)
    if existing_body == -1 or refreshed_body == -1:
        return refreshed
    return existing[:existing_body] + refreshed[refreshed_body:]


def _drain_events(job: DashboardJob) -> None:
    while True:
        try:
            event = job.events.get_nowait()
        except queue.Empty:
            return
        job.current_case = (event.case_id, event.repetition)
        job.actual_cost_usd = event.actual_reported_cost_usd
        if event.provider is not None:
            job.provider_current_case[event.provider] = (
                event.case_id,
                event.repetition,
            )
            if event.provider_elapsed_seconds is not None:
                job.provider_elapsed_seconds[event.provider] = (
                    event.provider_elapsed_seconds
                )
        if event.provider is not None and event.provider_completed_calls is not None:
            job.provider_completed_calls[event.provider] = event.provider_completed_calls
            if event.prediction is not None:
                job.provider_predictions[
                    (event.case_id, event.repetition, event.provider)
                ] = event.prediction
                cost = reported_cost(event.prediction)
                if cost is not None:
                    job.provider_cost_usd[event.provider] = (
                        job.provider_cost_usd.get(event.provider, 0.0) + cost
                    )
            if event.failure is not None:
                job.provider_failures[event.provider] = (
                    job.provider_failures.get(event.provider, 0) + 1
                )
        if event.result is not None:
            job.results[(event.case_id, event.repetition)] = event.result


PROVIDER_DISPLAY = {
    "jev": ("Jev", "#14B8A6"),
    "haiku": ("Claude Haiku", "#8B5CF6"),
    "sonnet": ("Claude Sonnet", "#F59E0B"),
}


def _apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 8% 0%, rgba(20,184,166,.10), transparent 30%),
                radial-gradient(circle at 92% 8%, rgba(139,92,246,.09), transparent 28%),
                #f7f9fc;
        }
        .block-container {max-width: 1450px; padding-top: 1.8rem; padding-bottom: 4rem;}
        .jev-hero {
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.35rem;
            border-radius: 20px;
            color: white;
            background: linear-gradient(120deg, #0f172a 0%, #164e63 55%, #115e59 100%);
            box-shadow: 0 14px 34px rgba(15,23,42,.14);
        }
        .jev-hero h1 {font-size: 2.15rem; line-height: 1.1; margin: 0 0 .45rem;}
        .jev-hero p {margin: 0; color: #d5f5ef; font-size: 1rem;}
        .jev-eyebrow {
            display: inline-block; margin-bottom: .55rem; color: #99f6e4;
            font-size: .74rem; font-weight: 750; letter-spacing: .12em; text-transform: uppercase;
        }
        .jev-model-title {font-size: 1.15rem; font-weight: 750; margin-bottom: .1rem;}
        .jev-model-id {color: #64748b; font-size: .76rem; min-height: 2.15rem; overflow-wrap: anywhere;}
        .jev-stat-grid {
            display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: .45rem; margin: .65rem 0 .8rem;
        }
        .jev-stat {
            min-width: 0; padding: .58rem .5rem; border: 1px solid #e2e8f0;
            border-radius: 11px; background: #f8fafc;
        }
        .jev-stat span {
            display: block; color: #64748b; font-size: .64rem; font-weight: 700;
            letter-spacing: .04em; text-transform: uppercase; white-space: nowrap;
        }
        .jev-stat strong {
            display: block; margin-top: .22rem; color: #172033;
            font-size: .92rem; line-height: 1.2; white-space: nowrap;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255,255,255,.90);
            border-color: #dbe4ee;
            border-radius: 16px;
            box-shadow: 0 8px 22px rgba(15,23,42,.06);
        }
        [data-testid="stMetric"] {
            background: rgba(255,255,255,.70);
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: .72rem .85rem;
        }
        [data-testid="stMetricLabel"] {color: #526174;}
        [data-testid="stDataFrame"] {border-radius: 14px; overflow: hidden;}
        .stButton > button {border-radius: 10px; font-weight: 650;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _provider_statistics(
    job: DashboardJob | None,
    provider: str,
) -> tuple[str, str, str, int, int]:
    if job is None:
        return "Pending", "Pending", "Pending", 0, 0
    provider_predictions = [
        (case_id, prediction)
        for (case_id, _repetition, prediction_provider), prediction in (
            job.provider_predictions.items()
        )
        if prediction_provider == provider
    ]
    if provider_predictions:
        gold_by_case = {case.id: case.gold_label for case in job.plan.cases}
        predictions = [prediction for _case_id, prediction in provider_predictions]
        quality = classification_metrics(
            (gold_by_case[case_id], prediction.label)
            for case_id, prediction in provider_predictions
        )
    else:
        predictions = [
            result.predictions[provider]
            for result in job.results.values()
            if provider in result.predictions
        ]
        quality = classification_metrics(
            (result.expected_label, result.predictions[provider].label)
            for result in job.results.values()
            if provider in result.predictions
        )
    accuracy = f"{quality.accuracy:.1%}" if quality.accuracy is not None else "Pending"
    macro_f1 = f"{quality.macro_f1:.1%}" if quality.macro_f1 is not None else "Pending"
    latency = (
        f"{mean(prediction.latency_ms for prediction in predictions):.0f} ms"
        if predictions
        else "Pending"
    )
    completed = job.provider_completed_calls.get(provider, len(predictions))
    failures = job.provider_failures.get(provider, 0)
    return accuracy, macro_f1, latency, completed, failures


def _render_model_comparison(
    plan: BenchmarkPlan,
    job: DashboardJob | None,
    execution_mode: str = SYNCHRONIZED_MODE,
) -> None:
    st.subheader("Model comparison")
    total = len(plan.cases) * plan.repetitions
    columns = st.columns(len(plan.models), gap="large")
    for column, provider in zip(columns, plan.models, strict=True):
        display_name, color = PROVIDER_DISPLAY.get(
            provider,
            (provider.title(), "#0EA5E9"),
        )
        accuracy, macro_f1, latency, completed, failures = _provider_statistics(
            job, provider
        )
        if job is None:
            cost = plan.estimated_cost_by_provider[provider] * total
            cost_label = "Estimated cost"
        else:
            cost = job.provider_cost_usd.get(provider, 0.0)
            cost_label = "Total cost" if job.done_event.is_set() else "Cost so far"
        with column.container(border=True):
            st.markdown(
                f'<div class="jev-model-title"><span style="color:{color}">&#9679;</span> '
                f"{display_name}</div>"
                f'<div class="jev-model-id">{plan.models[provider]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="jev-stat-grid">'
                f'<div class="jev-stat"><span>Accuracy</span><strong>{accuracy}</strong></div>'
                f'<div class="jev-stat"><span>Macro F1</span><strong>{macro_f1}</strong></div>'
                f'<div class="jev-stat"><span>Latency</span><strong>{latency}</strong></div>'
                f'<div class="jev-stat"><span>{cost_label}</span><strong>${cost:.6f}</strong></div>'
                "</div>",
                unsafe_allow_html=True,
            )
            st.progress(
                completed / total if total else 0,
                text=f"{completed} of {total} cases completed",
            )
            if failures:
                st.caption(f"{failures} provider failure{'s' if failures != 1 else ''}")
            elif job is None:
                st.caption("Estimate based on the latest cost-complete run")
            elif job.execution_mode == THROUGHPUT_MODE and completed:
                elapsed = job.provider_elapsed_seconds.get(provider)
                if elapsed and elapsed > 0:
                    st.caption(
                        f"{completed / elapsed * 60:.1f} cases/min · "
                        f"{elapsed:.1f}s elapsed"
                    )
                else:
                    st.caption("Independent model queue")
            else:
                st.caption("Successful and failed responses both count as completed")
    active_mode = job.execution_mode if job is not None else execution_mode
    if active_mode == THROUGHPUT_MODE:
        st.caption(
            "Throughput mode: each model advances to its next case independently, so "
            "the completion counters can diverge substantially."
        )
    else:
        st.caption(
            "Synchronized mode: all three models start each case in parallel, and the "
            "next case begins after all three finish."
        )


def _comparison_chart(
    rows: list[dict[str, object]],
    *,
    value_field: str,
    axis_title: str,
) -> alt.LayerChart:
    model_order = [str(row["model"]) for row in rows]
    colors = [str(row["color"]) for row in rows]
    maximum = max(float(row[value_field]) for row in rows)
    domain_max = maximum * 1.05 if maximum > 0 else 1.0
    chart_rows = [
        {
            **row,
            "ratio": float(row[value_field]) / maximum if maximum > 0 else 0.0,
        }
        for row in rows
    ]
    base = alt.Chart(alt.Data(values=chart_rows)).encode(
        y=alt.Y(
            "model:N",
            sort=model_order,
            axis=alt.Axis(title=None, labelLimit=105),
        ),
        x=alt.X(
            f"{value_field}:Q",
            scale=alt.Scale(domain=[0, domain_max]),
            axis=alt.Axis(title=axis_title, grid=True, tickCount=4),
        ),
    )
    bars = base.mark_bar(cornerRadiusEnd=6, size=24).encode(
        color=alt.Color(
            "model:N",
            scale=alt.Scale(domain=model_order, range=colors),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("model:N", title="Model"),
            alt.Tooltip(f"{value_field}:Q", title=axis_title),
        ],
    )
    inside_labels = base.transform_filter("datum.ratio >= 0.45").mark_text(
        align="right",
        baseline="middle",
        dx=-5,
        color="white",
        fontSize=10,
        fontWeight=700,
    ).encode(text=alt.Text("label:N"))
    outside_labels = base.transform_filter("datum.ratio < 0.45").mark_text(
        align="left",
        baseline="middle",
        dx=5,
        color="#334155",
        fontSize=11,
        fontWeight=600,
    ).encode(text=alt.Text("label:N"))
    return (bars + inside_labels + outside_labels).properties(
        height=145
    ).configure_view(stroke=None)


def _render_comparison_charts(job: DashboardJob) -> None:
    rows: list[dict[str, object]] = []
    for provider in job.plan.models:
        provider_predictions = [
            (case_id, prediction)
            for (case_id, _repetition, prediction_provider), prediction in (
                job.provider_predictions.items()
            )
            if prediction_provider == provider
        ]
        predictions = [prediction for _case_id, prediction in provider_predictions]
        if not predictions:
            provider_predictions = [
                (result.case_id, result.predictions[provider])
                for result in job.results.values()
                if provider in result.predictions
            ]
            predictions = [prediction for _case_id, prediction in provider_predictions]
        if not predictions:
            continue
        gold_by_case = {case.id: case.gold_label for case in job.plan.cases}
        quality = classification_metrics(
            (gold_by_case[case_id], prediction.label)
            for case_id, prediction in provider_predictions
        )
        display_name, color = PROVIDER_DISPLAY.get(
            provider,
            (provider.title(), "#0EA5E9"),
        )
        accuracy = quality.accuracy * 100 if quality.accuracy is not None else 0.0
        macro_f1 = quality.macro_f1 * 100 if quality.macro_f1 is not None else 0.0
        latency = mean(prediction.latency_ms for prediction in predictions)
        cost = job.provider_cost_usd.get(provider, 0.0)
        rows.append(
            {
                "model": display_name,
                "color": color,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "latency": latency,
                "cost": cost,
            }
        )
    if not rows:
        return

    st.subheader("Visual comparison")
    chart_columns = st.columns(4, gap="medium")
    chart_specs = (
        ("Accuracy", "Higher is better", "accuracy", "Accuracy (%)", ".1f"),
        ("Macro F1", "Higher is better", "macro_f1", "Macro F1 (%)", ".1f"),
        ("Mean latency", "Lower is better", "latency", "Milliseconds", ".0f"),
        ("Total model cost", "Lower is better", "cost", "USD", ".6f"),
    )
    for column, (title, guidance, field, axis_title, number_format) in zip(
        chart_columns,
        chart_specs,
        strict=True,
    ):
        chart_rows = [
            {
                **row,
                "label": (
                    f"${float(row[field]):{number_format}}"
                    if field == "cost"
                    else (
                        f"{float(row[field]):{number_format}}%"
                        if field in {"accuracy", "macro_f1"}
                        else f"{float(row[field]):{number_format}} ms"
                    )
                ),
            }
            for row in rows
        ]
        with column:
            st.markdown(f"#### {title}")
            st.caption(guidance)
            st.altair_chart(
                _comparison_chart(
                    chart_rows,
                    value_field=field,
                    axis_title=axis_title,
                ),
                width="stretch",
            )


def _render_classification_detail(job: DashboardJob) -> None:
    st.subheader("Per-label quality and confusion matrices")
    st.caption(
        "Precision, recall, and F1 are calculated over successful responses. "
        "Confusion-matrix rows are gold labels and columns are model predictions."
    )
    gold_by_case = {case.id: case.gold_label for case in job.plan.cases}
    columns = st.columns(len(job.plan.models), gap="large")
    for column, provider in zip(columns, job.plan.models, strict=True):
        provider_predictions = [
            (case_id, prediction)
            for (case_id, _repetition, prediction_provider), prediction in (
                job.provider_predictions.items()
            )
            if prediction_provider == provider
        ]
        if not provider_predictions:
            provider_predictions = [
                (result.case_id, result.predictions[provider])
                for result in job.results.values()
                if provider in result.predictions
            ]
        quality = classification_metrics(
            (gold_by_case[case_id], prediction.label)
            for case_id, prediction in provider_predictions
        )
        display_name, _color = PROVIDER_DISPLAY.get(
            provider,
            (provider.title(), "#0EA5E9"),
        )
        with column.container(border=True):
            st.markdown(f"#### {display_name}")
            st.caption(
                f"Macro F1: {quality.macro_f1:.1%} · "
                f"{quality.sample_count} successful predictions"
                if quality.macro_f1 is not None
                else "No successful predictions"
            )
            st.dataframe(
                [
                    {
                        "Label": label.value,
                        "Precision": f"{quality.per_label[label].precision:.1%}",
                        "Recall": f"{quality.per_label[label].recall:.1%}",
                        "F1": f"{quality.per_label[label].f1:.1%}",
                        "Support": quality.per_label[label].support,
                    }
                    for label in LABELS
                ],
                hide_index=True,
                width="stretch",
            )
            st.markdown("**Confusion matrix**")
            st.dataframe(
                [
                    {
                        "Actual": expected.value,
                        **{
                            predicted.value: quality.confusion_matrix[expected][
                                predicted
                            ]
                            for predicted in LABELS
                        },
                    }
                    for expected in LABELS
                ],
                hide_index=True,
                width="stretch",
            )


def _case_rows(job: DashboardJob) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for repetition in range(1, job.plan.repetitions + 1):
        for case in job.plan.cases:
            key = (case.id, repetition)
            result = job.results.get(key)
            if result is not None:
                status = "Complete"
            elif key == job.current_case and not job.done_event.is_set():
                status = "Running"
            else:
                status = "Pending"
            predictions = result.predictions if result is not None else {}
            failures = result.failures if result is not None else {}
            rows.append(
                {
                    "Case": case.id,
                    "Run": repetition,
                    "Status": status,
                    "Gold": case.gold_label.value,
                    "Jev": predictions["jev"].label.value
                    if "jev" in predictions
                    else ("FAIL" if "jev" in failures else ""),
                    "Haiku": predictions["haiku"].label.value
                    if "haiku" in predictions
                    else ("FAIL" if "haiku" in failures else ""),
                    "Sonnet": predictions["sonnet"].label.value
                    if "sonnet" in predictions
                    else ("FAIL" if "sonnet" in failures else ""),
                    "Disagreement": bool(
                        result and result.evaluation and result.evaluation.disagreement
                    ),
                    "Low confidence": ", ".join(
                        result.evaluation.low_confidence_providers
                        if result and result.evaluation
                        else []
                    ),
                }
            )
    return rows


def _render_provider_result(result: CaseResult, provider: str) -> None:
    st.markdown(f"#### {provider.title()}")
    if provider in result.failures:
        failure = result.failures[provider]
        st.error(f"{failure.error_type}: {failure.message}")
        return
    prediction = result.predictions.get(provider)
    if prediction is None:
        st.write("Pending")
        return
    metric_columns = st.columns(3)
    metric_columns[0].metric("Decision", prediction.label.value)
    metric_columns[1].metric(
        "Confidence",
        "n/a" if prediction.confidence is None else f"{prediction.confidence:.3f}",
    )
    metric_columns[2].metric("Latency", f"{prediction.latency_ms:.0f} ms")
    probabilities = prediction.probabilities or {}
    for label in Ownership:
        probability = probabilities.get(label)
        if probability is None:
            st.caption(f"{label.value}: n/a")
        else:
            st.progress(probability, text=f"{label.value}: {probability:.1%}")
    cost = reported_cost(prediction)
    st.caption(
        f"Cost: {'n/a' if cost is None else f'${cost:.6f}'} · "
        f"Retries: {prediction.retry_count} · Model: {prediction.served_model or prediction.requested_model}"
    )


def _render_case_detail(job: DashboardJob) -> None:
    completed_keys = list(job.results)
    if not completed_keys:
        return
    st.subheader("Inspect one case")
    selected = st.selectbox(
        "Case",
        completed_keys,
        index=len(completed_keys) - 1,
        format_func=lambda key: f"{key[0]} · run {key[1]}",
    )
    result = job.results[selected]
    case = next(case for case in job.plan.cases if case.id == selected[0])
    st.caption(
        f"Gold label: {case.gold_label.value} · Reason: {case.reason_code} · "
        f"Difficulty: {case.difficulty}"
    )
    if st.toggle(
        "Show case transcript",
        key=f"show-transcript-{selected[0]}-{selected[1]}",
    ):
        st.text(case.case_text)
    columns = st.columns(3)
    for column, provider in zip(columns, ("jev", "haiku", "sonnet"), strict=True):
        with column:
            _render_provider_result(result, provider)


def _clear_job_and_rerun() -> None:
    st.session_state.pop("benchmark_job", None)
    st.rerun()


def _initialize_run_settings() -> None:
    if "dashboard_dataset" not in st.session_state:
        st.session_state["dashboard_dataset"] = "data/seeds/cases.csv"
    if "dashboard_repetitions" not in st.session_state:
        st.session_state["dashboard_repetitions"] = 1
    if "dashboard_execution_mode" not in st.session_state:
        st.session_state["dashboard_execution_mode"] = "Synchronized"


def _render_run_settings() -> None:
    st.subheader("Run settings")
    with st.container(border=True):
        fields = st.columns([2, 1])
        fields[0].text_input("Dataset", key="dashboard_dataset")
        fields[1].number_input(
            "Repetitions",
            min_value=1,
            max_value=10,
            step=1,
            key="dashboard_repetitions",
        )
        st.radio(
            "Execution mode",
            ("Synchronized", "Throughput"),
            captions=(
                "Each case waits for all models.",
                "Each model advances independently.",
            ),
            horizontal=True,
            key="dashboard_execution_mode",
        )


@st.fragment(run_every="1s")
def _render_live_monitor() -> None:
    job: DashboardJob | None = st.session_state.get("benchmark_job")
    if job is None:
        return
    _drain_events(job)
    total = len(job.plan.cases) * job.plan.repetitions
    completed = len(job.results)
    if job.thread is None and job.execution is not None:
        st.caption(f"Viewing saved run: {job.execution.output_path}")
    _render_model_comparison(job.plan, job)
    if job.done_event.is_set() and job.execution is not None:
        _render_comparison_charts(job)
        _render_classification_detail(job)
    st.divider()
    if job.error:
        st.error(job.error)
    elif job.done_event.is_set() and job.execution is not None:
        if job.execution.stopped_reason:
            st.warning(f"Stopped: {job.execution.stopped_reason}")
        else:
            st.success(f"Benchmark complete: {completed} cases evaluated")
    else:
        st.subheader("Run progress")
        st.progress(
            completed / total if total else 0,
            text=f"{completed} of {total} case runs complete",
        )
        if job.execution_mode == THROUGHPUT_MODE and job.provider_current_case:
            queue_status = " · ".join(
                f"{PROVIDER_DISPLAY.get(provider, (provider.title(), ''))[0]}: {case_id}"
                for provider, (case_id, _repetition) in sorted(
                    job.provider_current_case.items()
                )
            )
            st.info(f"Independent queues · {queue_status}")
            if st.button("Stop after active calls", type="secondary"):
                job.stop_event.set()
                st.warning("Stop requested. Active calls will finish and be saved.")
        elif job.current_case:
            st.info(
                f"Running {job.current_case[0]} · repetition {job.current_case[1]}"
            )
            if st.button("Stop after current case", type="secondary"):
                job.stop_event.set()
                st.warning("Stop requested. The current case will finish and be saved.")
        else:
            st.info("Starting benchmark")
        with st.expander("Budget protection", expanded=False):
            budget_columns = st.columns(2)
            budget_columns[0].metric(
                "Reported spend so far",
                f"${job.actual_cost_usd:.6f}",
            )
            if job.maximum_cost_usd is not None:
                budget_columns[1].metric(
                    "Cost limit",
                    f"${job.maximum_cost_usd:.2f}",
                )
            st.caption(
                "The cost limit is a safety guard for launching the next case; "
                "the model cards above are the comparison view."
            )

    with st.expander("Case-level results and diagnostics", expanded=False):
        st.caption(
            "Optional detail for investigating disagreements, low confidence, or an "
            "unexpected model decision."
        )
        st.dataframe(_case_rows(job), hide_index=True, width="stretch")
        _render_case_detail(job)

    if job.done_event.is_set() and job.execution is not None:
        download_columns = st.columns(2)
        if job.execution.output_path.exists():
            download_columns[0].download_button(
                "Download raw JSON",
                data=job.execution.output_path.read_bytes(),
                file_name=job.execution.output_path.name,
                mime="application/json",
            )
        if job.execution.report_path.exists():
            download_columns[1].download_button(
                "Download Markdown report",
                data=_downloadable_report(job),
                file_name=job.execution.report_path.name,
                mime="text/markdown",
            )
        st.button("Prepare another run", on_click=_clear_job_and_rerun)


def main() -> None:
    st.set_page_config(page_title="Case Ownership Routing", layout="wide")
    _apply_theme()
    st.markdown(
        """
        <div class="jev-hero">
          <div class="jev-eyebrow">Model evaluation workspace</div>
          <h1>Case Ownership Routing</h1>
          <p>Compare Jev, Claude Haiku, and Claude Sonnet on approved ownership cases.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    job: DashboardJob | None = st.session_state.get("benchmark_job")
    if job is None:
        _initialize_run_settings()
        dataset_text = str(st.session_state["dashboard_dataset"])
        repetitions = int(st.session_state["dashboard_repetitions"])
        execution_mode = (
            THROUGHPUT_MODE
            if st.session_state["dashboard_execution_mode"] == "Throughput"
            else SYNCHRONIZED_MODE
        )
    else:
        dataset_text = str(job.plan.dataset_path)
        repetitions = job.plan.repetitions
        execution_mode = job.execution_mode

    plan: BenchmarkPlan | None = job.plan if job else None
    plan_error: str | None = None
    if plan is None:
        try:
            plan = prepare_benchmark_plan(
                dataset_path=Path(dataset_text),
                repetitions=int(repetitions),
            )
        except (OSError, ValueError) as exc:
            plan_error = str(exc)

    if plan_error:
        if job is None:
            _render_run_settings()
        st.error(plan_error)
        return
    if plan is None:
        st.error("Unable to prepare benchmark plan")
        return

    if job is None:
        _render_model_comparison(plan, None, execution_mode)
        st.divider()
        _render_run_settings()
        st.subheader("Run preview")
        summary = st.columns(2)
        summary[0].metric("Approved cases", len(plan.cases))
        summary[1].metric("Planned calls", plan.planned_calls)
        latest_saved = _latest_saved_benchmark()
        if latest_saved is not None:
            if st.button(
                f"View latest saved run · {latest_saved.stem.removeprefix('benchmark-')}",
                type="secondary",
            ):
                try:
                    st.session_state.benchmark_job = _load_saved_job(plan, latest_saved)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    st.error(str(exc))
                else:
                    st.rerun()
        with st.expander(
            "Start a paid run · budget protection",
            expanded=False,
        ):
            budget_summary = st.columns(2)
            budget_summary[0].metric(
                "Estimated total",
                f"${plan.estimated_cost_usd:.3f}",
            )
            budget_summary[1].metric(
                "Guarded estimate",
                f"${plan.guarded_estimate_usd:.3f}",
            )
            st.caption(
                f"Estimate source: {plan.estimate_source} · "
                f"Retry reserve: {plan.retry_reserve:.0%}"
            )
            default_limit = math.ceil(plan.guarded_estimate_usd * 100) / 100
            maximum_cost = st.number_input(
                "Maximum cost (USD)",
                min_value=0.01,
                value=max(0.01, default_limit),
                step=0.01,
                format="%.2f",
            )
            understands_billing = st.checkbox(
                "I understand that starting the live benchmark makes billable API calls."
            )
            within_budget = maximum_cost >= plan.guarded_estimate_usd
            if not within_budget:
                st.warning(
                    "The cost limit is below the guarded estimate. The run cannot start."
                )
            if st.button(
                "Start live benchmark",
                type="primary",
                disabled=not understands_billing or not within_budget,
            ):
                st.session_state.benchmark_job = _start_job(
                    plan,
                    maximum_cost,
                    execution_mode,
                )
                st.rerun()
            st.info(
                "No API calls occur until billing is acknowledged and the run is started."
            )
        return

    _render_live_monitor()


if __name__ == "__main__":
    main()
