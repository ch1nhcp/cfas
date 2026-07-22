"""End-to-end pipeline: classify -> retrieve -> gate -> report.

Error policy:
- Transient LLM API errors are retried by RetryingClient (backoff).
- Schema failures surface as ClassificationError/ReportGenerationError
  after their single repair retry.
- When any of those is terminal, make_failure_report() produces a valid
  FeedbackReport with status=processing_failed routed straight to a human -
  the pipeline never returns nothing.
- Local tool errors are NOT exceptions: they flow through the retrieval
  state machine into missing_context and forced review.

Every run yields a trace with all intermediate steps.
"""

import time
import uuid
from collections.abc import Callable

import anthropic
from pydantic import BaseModel, ConfigDict

from cfas.agent import gather_context
from cfas.classify import ClassificationError, classify_feedback
from cfas.gate import validate_context
from cfas.llm import default_client
from cfas.models import FeedbackReport, FeedbackSubmission
from cfas.report import ReportGenerationError, generate_report, make_failure_report
from cfas.retry import RetryingClient
from cfas.tools import DATA_DIR
from cfas.trace import TraceRecorder

# Terminal pipeline failures -> failure report. anthropic.APIError covers
# both connection errors (after retries) and non-retryable API statuses.
PIPELINE_ERRORS = (ClassificationError, ReportGenerationError, anthropic.APIError)

# Configuration/deployment errors must crash loudly instead of being
# converted into per-item manual-triage reports: an invalid API key or a
# malformed request is an operator problem, not a feedback-processing
# outcome.
CONFIG_ERRORS = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.NotFoundError,
    anthropic.BadRequestError,
    anthropic.UnprocessableEntityError,
)


class PipelineResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    report: FeedbackReport
    trace: dict


def process_feedback(
    submission: FeedbackSubmission,
    client=None,
    data_dir=DATA_DIR,
    sleep: Callable[[float], None] = time.sleep,
) -> PipelineResult:
    """Process one submission end-to-end. Always returns a valid report -
    on terminal failure, a processing_failed report for manual triage."""
    run_id = f"RUN-{uuid.uuid4().hex[:10]}"
    report_id = f"RPT-{uuid.uuid4().hex[:10]}"
    recorder = TraceRecorder(run_id)
    retrying = RetryingClient(client or default_client(), sleep=sleep)

    recorder.record("intake", submission=submission.model_dump(mode="json"))
    classification = None
    try:
        classification = classify_feedback(submission, client=retrying)
        recorder.record(
            "classification", result=classification.model_dump(mode="json")
        )

        retrieval = gather_context(
            submission, classification, client=retrying, data_dir=data_dir
        )
        recorder.record(
            "retrieval",
            iterations=retrieval.iterations,
            source_states={
                s.value: st.value for s, st in retrieval.source_states.items()
            },
            tool_calls=[r.model_dump() for r in retrieval.tool_calls],
            warnings=retrieval.warnings,
        )

        context_validation = validate_context(classification, retrieval)
        recorder.record(
            "validate_context",
            review_reasons=context_validation.review_reasons,
            missing_context=context_validation.missing_context,
        )

        report = generate_report(
            classification=classification,
            retrieval=retrieval,
            context_validation=context_validation,
            client=retrying,
            report_id=report_id,
        )
        recorder.record(
            "report",
            confidence=report.confidence,
            needs_human_review=report.needs_human_review,
            review_reason=report.review_reason,
            warnings=report.warnings,
        )
    except PIPELINE_ERRORS as exc:
        if isinstance(exc, CONFIG_ERRORS):
            raise
        error_summary = f"{type(exc).__name__}: {exc}"
        recorder.record("processing_failed", error=error_summary)
        report = make_failure_report(report_id, error_summary, classification)

    recorder.record(
        "final", status=report.status.value, retry_count=retrying.retry_count
    )
    return PipelineResult(report=report, trace=recorder.to_dict())
