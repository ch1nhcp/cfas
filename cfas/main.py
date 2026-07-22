"""CLI entry point.

Usage:
    python -m cfas.main "The app crashed after the last update" \
        --customer-id CUST-001 --channel email
"""

import argparse
import sys

from pydantic import ValidationError

from cfas.intake import build_submission
from cfas.models import Channel


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
        default=Channel.WEB_FORM.value,
        choices=[c.value for c in Channel],
        help="Channel the feedback arrived through",
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

    # Pipeline stages (classification, retrieval, report) are wired in here
    # as they land. For now intake echoes the validated submission.
    print(submission.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
