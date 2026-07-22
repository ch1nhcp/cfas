"""Observable trace: every run records its intermediate steps.

Includes decision rationale (each tool call's required `reason`), tool
arguments/statuses/source IDs, validation warnings, retry count, and final
status - the reviewer can see HOW the report was reached, not just the
final output.

Production note (for the write-up): PII would be redacted and raw reasoning
not stored long-term; in this assessment demo the reasoning steps are an
explicit requirement.
"""

import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


class TraceRecorder:
    """Append-only step recorder for one pipeline run."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.started_at = datetime.now(timezone.utc)
        self._t0 = time.monotonic()
        self._steps: list[dict] = []

    def record(self, step: str, **data) -> None:
        self._steps.append(
            {
                "step": step,
                "elapsed_ms": int((time.monotonic() - self._t0) * 1000),
                **data,
            }
        )

    def to_dict(self) -> dict:
        # deep copy: callers must not be able to mutate recorded history
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "steps": copy.deepcopy(self._steps),
        }


def print_trace_summary(trace: dict, stream=None) -> None:
    """Human-readable console rendering of a trace."""
    stream = stream or sys.stdout
    print(f"=== run {trace['run_id']} ({trace['started_at']}) ===", file=stream)
    for step in trace["steps"]:
        details = {k: v for k, v in step.items() if k not in ("step", "elapsed_ms")}
        rendered = json.dumps(details, ensure_ascii=False, default=str)
        if len(rendered) > 300:
            rendered = rendered[:297] + "..."
        print(f"[{step['elapsed_ms']:>6}ms] {step['step']}: {rendered}", file=stream)


def write_run_artifacts(
    run_id: str, submission_json: dict, trace: dict, report_json: dict, out_dir: Path
) -> Path:
    """Write input.json / trace.json / report.json under out_dir/<run_id>/."""
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("input.json", submission_json),
        ("trace.json", trace),
        ("report.json", report_json),
    ):
        with open(run_dir / name, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return run_dir
