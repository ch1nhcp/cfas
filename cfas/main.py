"""CLI entry point: process one feedback submission end-to-end.

Usage:
    python -m cfas.main "The app crashed after the last update" \
        --customer-id CUST-001 --channel email

Failure injection (for demonstrating error handling without breaking
anything real):
    --inject-failure llm    every LLM call raises a connection error
    --inject-failure tool   tools read from a nonexistent data directory
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import anthropic
from pydantic import ValidationError

from cfas.intake import build_submission
from cfas.models import Channel, ReportStatus
from cfas.pipeline import CONFIG_ERRORS, process_feedback
from cfas.tools import DATA_DIR
from cfas.trace import print_trace_summary, write_run_artifacts


class InjectedLLMFailure(anthropic.APIConnectionError):
    """Simulated transient LLM failure; bypasses the SDK constructor on
    purpose - only the exception type matters to the retry policy."""

    def __init__(self):  # noqa: D107
        Exception.__init__(self, "injected LLM failure")


class _FailingLLMClient:
    """Client whose every call fails - exercises retry + failure report."""

    def __init__(self):
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **_kwargs):
        raise InjectedLLMFailure()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cfas",
        description="Agentic Customer Feedback System - process one feedback submission.",
    )
    parser.add_argument("feedback_text", help="Free-text customer feedback")
    parser.add_argument(
        "--customer-id",
        default=None,
        help="Customer ID (e.g. CUST-001); omit for anonymous feedback",
    )
    parser.add_argument(
        "--channel",
        required=True,
        choices=[c.value for c in Channel],
        help="Channel the feedback arrived through",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs"),
        help="Directory for run artifacts (input/trace/report JSON)",
    )
    parser.add_argument(
        "--inject-failure",
        choices=["llm", "tool"],
        default=None,
        help="Simulate a failure mode for demonstration",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        submission = build_submission(
            feedback_text=args.feedback_text,
            customer_id=args.customer_id,
            channel=args.channel,
        )
    except ValidationError as exc:
        print("Invalid submission:", file=sys.stderr)
        for error in exc.errors():
            field = ".".join(str(part) for part in error["loc"]) or "input"
            print(f"  - {field}: {error['msg']}", file=sys.stderr)
        return 1

    client = _FailingLLMClient() if args.inject_failure == "llm" else None
    data_dir = (
        Path("/nonexistent-injected-tool-failure")
        if args.inject_failure == "tool"
        else DATA_DIR
    )

    try:
        result = process_feedback(submission, client=client, data_dir=data_dir)
    except CONFIG_ERRORS as exc:
        # operator problem (bad key, malformed request), not a feedback
        # outcome - crash loudly instead of writing a failure report
        print(f"Configuration error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print_trace_summary(result.trace, stream=sys.stderr)
    run_dir = write_run_artifacts(
        run_id=result.trace["run_id"],
        submission_json=submission.model_dump(mode="json"),
        trace=result.trace,
        report_json=result.report.model_dump(mode="json"),
        out_dir=args.out_dir,
    )
    print(f"artifacts: {run_dir}", file=sys.stderr)
    print(result.report.model_dump_json(indent=2))
    # 0: report produced; 3: pipeline fell back to a processing_failed
    # report - scripts can distinguish without parsing the JSON
    return 3 if result.report.status is ReportStatus.PROCESSING_FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
