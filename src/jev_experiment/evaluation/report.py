"""Human-readable reporting for the initial three-case smoke run."""

from collections.abc import Iterable
from statistics import mean

from jev_experiment.evaluation.metrics import LABELS, classification_metrics
from jev_experiment.models import CaseResult, Prediction

PROVIDERS = ("jev", "haiku", "sonnet")


def _format_number(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _mean(values: Iterable[float]) -> float | None:
    collected = list(values)
    return mean(collected) if collected else None


def _predictions_for(results: list[CaseResult], provider: str) -> list[Prediction]:
    return [result.predictions[provider] for result in results if provider in result.predictions]


def _quality_for(results: list[CaseResult], provider: str):
    return classification_metrics(
        (result.expected_label, result.predictions[provider].label)
        for result in results
        if provider in result.predictions
    )


def reported_cost(prediction: Prediction) -> float | None:
    """Read normalized cost, falling back to retained OpenRouter response metadata."""

    if prediction.usage.cost_usd is not None:
        return prediction.usage.cost_usd
    raw = prediction.raw_response
    if not isinstance(raw, dict):
        return None
    debug = raw.get("debug")
    if not isinstance(debug, dict) or not isinstance(debug.get("llm_attempts"), list):
        return None
    costs: list[float] = []
    for attempt in debug["llm_attempts"]:
        if not isinstance(attempt, dict):
            continue
        response = attempt.get("llm_response")
        if not isinstance(response, dict):
            continue
        usage = response.get("usage")
        if isinstance(usage, dict) and isinstance(usage.get("cost"), (int, float)):
            costs.append(float(usage["cost"]))
    return sum(costs) if costs else None


def render_smoke_report(results: list[CaseResult]) -> str:
    """Render deterministic summary statistics and case-level outcomes."""

    lines = [
        "# Provider Smoke-Test Report",
        "",
        f"Cases: {len(results)}. Expected provider calls: {len(results) * len(PROVIDERS)}.",
        "",
        "## Provider summary",
        "",
        "| Provider | Successes | Failures | Accuracy* | Macro F1* | Mean latency | Mean confidence | Input tokens | Output tokens | Reported cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for provider in PROVIDERS:
        predictions = _predictions_for(results, provider)
        quality = _quality_for(results, provider)
        successes = len(predictions)
        failures = len(results) - successes
        accuracy = quality.accuracy * 100 if quality.accuracy is not None else None
        latency = _mean(prediction.latency_ms for prediction in predictions)
        confidence = _mean(
            prediction.confidence
            for prediction in predictions
            if prediction.confidence is not None
        )
        input_tokens = sum(
            prediction.usage.input_tokens or 0 for prediction in predictions
        )
        output_tokens = sum(
            prediction.usage.output_tokens or 0 for prediction in predictions
        )
        reported_costs = [cost for prediction in predictions if (cost := reported_cost(prediction)) is not None]
        cost_text = f"${sum(reported_costs):.6f}" if len(reported_costs) == successes else "n/a"
        lines.append(
            f"| {provider} | {successes} | {failures} | "
            f"{_format_number(accuracy)}% | {_format_number(quality.macro_f1, 3)} | "
            f"{_format_number(latency)} ms | "
            f"{_format_number(confidence, 3)} | {input_tokens} | {output_tokens} | {cost_text} |"
        )

    disagreement_count = sum(
        bool(result.evaluation and result.evaluation.disagreement) for result in results
    )
    all_predictions = [
        prediction for result in results for prediction in result.predictions.values()
    ]
    all_reported_costs = [
        cost for prediction in all_predictions if (cost := reported_cost(prediction)) is not None
    ]
    missing_cost_providers = sorted(
        {
            prediction.provider
            for prediction in all_predictions
            if reported_cost(prediction) is None
        }
    )
    if not all_predictions:
        cost_coverage = "no successful calls to measure"
    elif missing_cost_providers:
        cost_coverage = (
            f"{len(all_reported_costs)} of {len(all_predictions)} successful calls; "
            f"missing for: {', '.join(missing_cost_providers)}"
        )
    else:
        cost_coverage = (
            f"{len(all_reported_costs)} of {len(all_predictions)} successful calls (complete)"
        )
    lines.extend(
        [
            "",
            "*Accuracy and Macro F1 are calculated over successful responses only; failures are shown separately.",
            "",
            "## Per-label quality",
            "",
            "Precision, recall, and F1 use a 0-to-1 scale. Support is the number of gold-label cases.",
            "",
            "| Provider | Label | Precision | Recall | F1 | Support |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for provider in PROVIDERS:
        quality = _quality_for(results, provider)
        for label in LABELS:
            metrics = quality.per_label[label]
            lines.append(
                f"| {provider} | {label.value} | {metrics.precision:.3f} | "
                f"{metrics.recall:.3f} | {metrics.f1:.3f} | {metrics.support} |"
            )

    lines.extend(["", "## Confusion matrices", ""])
    for provider in PROVIDERS:
        quality = _quality_for(results, provider)
        lines.extend(
            [
                f"### {provider}",
                "",
                "Rows are gold labels; columns are model predictions.",
                "",
                "| Actual \\ Predicted | MINE | NOT_MINE | UNSURE |",
                "|---|---:|---:|---:|",
            ]
        )
        for expected in LABELS:
            counts = quality.confusion_matrix[expected]
            lines.append(
                f"| {expected.value} | {counts[LABELS[0]]} | "
                f"{counts[LABELS[1]]} | {counts[LABELS[2]]} |"
            )
        lines.append("")

    lines.extend(
        [
            "",
            "## Run summary",
            "",
            f"- Cases with provider disagreement: {disagreement_count} of {len(results)}",
            f"- Successful calls: {sum(len(result.predictions) for result in results)}",
            f"- Failed calls: {sum(len(result.failures) for result in results)}",
            f"- Provider-reported cost captured: ${sum(all_reported_costs):.6f}",
            f"- Cost coverage: {cost_coverage}.",
            "",
            "## Case results",
            "",
            "| Case | Expected | Jev | Haiku | Sonnet | Disagreement |",
            "|---|---|---|---|---|---|",
        ]
    )
    for result in results:
        case_display = (
            f"{result.case_id} (run {result.repetition})"
            if result.repetition > 1
            else result.case_id
        )
        labels = [
            result.predictions[provider].label.value
            if provider in result.predictions
            else "FAILURE"
            for provider in PROVIDERS
        ]
        disagreement = "yes" if result.evaluation and result.evaluation.disagreement else "no"
        lines.append(
            f"| {case_display} | {result.expected_label.value} | "
            f"{labels[0]} | {labels[1]} | {labels[2]} | {disagreement} |"
        )

    lines.extend(
        [
            "",
            "## Case confidence and probabilities",
            "",
            "| Case | Provider | Label | Confidence | P(MINE) | P(NOT_MINE) | P(UNSURE) | Low confidence |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for result in results:
        case_display = (
            f"{result.case_id} (run {result.repetition})"
            if result.repetition > 1
            else result.case_id
        )
        low_confidence = set(
            result.evaluation.low_confidence_providers
            if result.evaluation is not None
            else []
        )
        for provider in PROVIDERS:
            prediction = result.predictions.get(provider)
            if prediction is None:
                lines.append(
                    f"| {case_display} | {provider} | FAILURE | n/a | n/a | n/a | n/a | n/a |"
                )
                continue
            probabilities = prediction.probabilities or {}
            lines.append(
                f"| {case_display} | {provider} | {prediction.label.value} | "
                f"{_format_number(prediction.confidence, 3)} | "
                f"{_format_number(probabilities.get('MINE'), 3)} | "
                f"{_format_number(probabilities.get('NOT_MINE'), 3)} | "
                f"{_format_number(probabilities.get('UNSURE'), 3)} | "
                f"{'yes' if provider in low_confidence else 'no'} |"
            )

    failures = [
        (result.case_id, failure)
        for result in results
        for failure in result.failures.values()
    ]
    if failures:
        lines.extend(["", "## Failures", ""])
        for case_id, failure in failures:
            message = " ".join(failure.message.splitlines())
            lines.append(
                f"- `{case_id}` / `{failure.provider}`: `{failure.error_type}` — {message}"
            )

    lines.append("")
    return "\n".join(lines)
