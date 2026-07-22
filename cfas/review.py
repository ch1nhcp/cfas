"""Human-in-the-loop review stub.

Every report starts as pending_review; no suggested action is executed
until a CS officer decides:

    pending_review -> approved | overridden | rejected

- approve: the suggested actions stand as-is.
- override: the officer supplies replacement actions (required, non-empty).
- reject: the report is discarded from the action queue.

processing_failed reports are not reviewable here - they carry no
trustworthy suggestion to approve and are routed to manual triage instead.

Production design note (for the write-up): this would be a queue + audit
log - reports land in a review inbox ordered by urgency/needs_human_review,
decisions are persisted with reviewer identity and timestamp, overrides
feed back into an evaluation set, and only approved/overridden actions are
dispatched to downstream systems (ticketing, refunds) with RBAC on who may
approve which action types.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from cfas.models import FeedbackReport, ReportStatus, SuggestedAction


class ReviewDecision(StrEnum):
    APPROVE = "approve"
    OVERRIDE = "override"
    REJECT = "reject"


_STATUS_BY_DECISION = {
    ReviewDecision.APPROVE: ReportStatus.APPROVED,
    ReviewDecision.OVERRIDE: ReportStatus.OVERRIDDEN,
    ReviewDecision.REJECT: ReportStatus.REJECTED,
}


class ReviewError(Exception):
    """Invalid review transition or parameters."""


class ReviewRecord(BaseModel):
    """Audit record of one review decision."""

    model_config = ConfigDict(frozen=True)

    report_id: str
    decision: ReviewDecision
    reviewer_id: str
    note: str | None
    resulting_status: ReportStatus


def apply_review(
    report: FeedbackReport,
    decision: ReviewDecision,
    reviewer_id: str,
    note: str | None = None,
    overridden_actions: list[SuggestedAction] | None = None,
) -> tuple[FeedbackReport, ReviewRecord]:
    """Apply a CS officer's decision. Returns (updated report, audit record);
    the input report is never mutated."""
    try:
        # normalize: a raw string like "override" must behave identically
        # to the enum member (identity checks below rely on it)
        decision = ReviewDecision(decision)
    except ValueError as exc:
        raise ReviewError(f"unknown decision '{decision}'") from exc
    if report.status is not ReportStatus.PENDING_REVIEW:
        raise ReviewError(
            f"only pending_review reports can be reviewed "
            f"(got '{report.status.value}')"
        )
    if not reviewer_id.strip():
        raise ReviewError("reviewer_id must be a non-empty identifier")

    if decision is ReviewDecision.OVERRIDE:
        if not overridden_actions:
            raise ReviewError("override requires non-empty overridden_actions")
    elif overridden_actions is not None:
        raise ReviewError(
            "overridden_actions is only valid with the override decision"
        )

    new_status = _STATUS_BY_DECISION[decision]
    updates: dict = {"status": new_status}
    if decision is ReviewDecision.OVERRIDE:
        updates["suggested_actions"] = list(overridden_actions)

    updated = report.model_copy(update=updates)
    record = ReviewRecord(
        report_id=report.report_id,
        decision=decision,
        reviewer_id=reviewer_id.strip(),
        note=note,
        resulting_status=new_status,
    )
    return updated, record
