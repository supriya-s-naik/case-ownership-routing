"""Command-line entry point."""

import argparse
import subprocess
import sys
from pathlib import Path

from jev_experiment.benchmark import (
    format_benchmark_plan,
    prepare_benchmark_plan,
    run_live_benchmark_sync,
)
from jev_experiment.dataset import (
    load_cases,
    load_reason_codes,
    validate_case_collection,
    write_cases_csv,
)
from jev_experiment.experiment import render_saved_smoke_report, run_live_smoke_sync


def main() -> None:
    parser = argparse.ArgumentParser(prog="jev-experiment")
    subparsers = parser.add_subparsers(dest="command")
    smoke = subparsers.add_parser("smoke", help="Run the three-case provider smoke test")
    smoke.add_argument(
        "--live",
        action="store_true",
        help="Confirm that billable OpenRouter requests may be made",
    )
    report = subparsers.add_parser("report", help="Regenerate a report without API calls")
    report.add_argument("results_path", type=Path, help="Saved smoke JSON file")
    validate = subparsers.add_parser("validate-data", help="Validate a JSONL case dataset")
    validate.add_argument("dataset_path", type=Path, help="JSONL or CSV dataset to validate")
    export_csv = subparsers.add_parser("export-csv", help="Export cases as an editable CSV")
    export_csv.add_argument("source_path", type=Path, help="Source JSONL or CSV dataset")
    export_csv.add_argument("output_path", type=Path, help="Destination CSV file")
    benchmark = subparsers.add_parser(
        "benchmark",
        help="Preview or run an approved dataset with a cost limit",
    )
    benchmark.add_argument("dataset_path", type=Path, help="Approved JSONL or CSV dataset")
    benchmark.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of runs per case (default: 1)",
    )
    benchmark.add_argument(
        "--estimate-from",
        type=Path,
        help="Saved result JSON used to estimate provider costs",
    )
    benchmark.add_argument(
        "--live",
        action="store_true",
        help="Confirm that billable OpenRouter requests may be made",
    )
    benchmark.add_argument(
        "--max-cost-usd",
        type=float,
        help="Required live-run spending limit in USD",
    )
    subparsers.add_parser("ui", help="Launch the local benchmark dashboard")
    args = parser.parse_args()

    if args.command == "report":
        print(f"Summary report: {render_saved_smoke_report(args.results_path)}")
        return
    if args.command == "validate-data":
        summary = validate_case_collection(load_cases(args.dataset_path), load_reason_codes())
        print(summary.model_dump_json(indent=2))
        return
    if args.command == "export-csv":
        cases = load_cases(args.source_path)
        write_cases_csv(cases, args.output_path)
        print(f"Exported {len(cases)} cases to {args.output_path}")
        return
    if args.command == "benchmark":
        try:
            plan = prepare_benchmark_plan(
                args.dataset_path,
                repetitions=args.repetitions,
                estimate_from=args.estimate_from,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(format_benchmark_plan(plan))
        if not args.live:
            print("\nPreview only. Add --live and --max-cost-usd to authorize billing.")
            return
        if args.max_cost_usd is None:
            parser.error("benchmark --live requires --max-cost-usd")
        if args.max_cost_usd <= 0:
            parser.error("--max-cost-usd must be positive")
        if plan.guarded_estimate_usd > args.max_cost_usd:
            parser.error(
                f"budget-guarded estimate ${plan.guarded_estimate_usd:.6f} "
                f"exceeds the ${args.max_cost_usd:.6f} limit; no API calls were made"
            )
        execution = run_live_benchmark_sync(
            plan,
            maximum_cost_usd=args.max_cost_usd,
        )
        status = "stopped early" if execution.stopped_reason else "complete"
        print(f"\nBenchmark {status}: {len(execution.results)} case runs")
        print(f"Actual provider-reported cost: ${execution.actual_reported_cost_usd:.6f}")
        if execution.stopped_reason:
            print(f"Stop reason: {execution.stopped_reason}")
        print(f"Raw results: {execution.output_path}")
        print(f"Summary report: {execution.report_path}")
        return
    if args.command == "ui":
        dashboard_path = Path(__file__).with_name("dashboard.py")
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
            check=True,
        )
        return
    if args.command != "smoke":
        parser.print_help()
        return
    if not args.live:
        parser.error("smoke requires --live because it makes billable API requests")

    results, output, report_output = run_live_smoke_sync()
    successes = sum(len(result.predictions) for result in results)
    failures = sum(len(result.failures) for result in results)
    print(f"Smoke run complete: {successes} predictions, {failures} failures")
    print(f"Raw results: {output}")
    print(f"Summary report: {report_output}")


if __name__ == "__main__":
    main()
