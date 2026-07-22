"""Human-review stub tests: transitions and invariants."""

import pytest
from factories import make_action, make_classification, make_draft

from cfas.models import ActionType, FeedbackReport, ReportStatus, SuggestedAction
from cfas.review import ReviewDecision, ReviewError, apply_review


def make_report(status=ReportStatus.PENDING_REVIEW, **overrides):
    draft = make_draft()
    fields = {
        "report_id": "RPT-0001",
        **draft.model_dump(),
        "classification": make_classification(),
        "warnings": [],
        "missing_context": [],
        "needs_human_review": False,
        "review_reason": None,
        "status": status,
    }
    return FeedbackReport(**{**fields, **overrides})


OVERRIDE_ACTIONS = [
    SuggestedAction(
        action_type=ActionType.CUSTOMER_RESPONSE,
        action="Apologize and offer account credit instead.",
        source_ids=["POL-REFUND-001"],
    )
]


class TestApprove:
    def test_approve_transitions_to_approved(self):
        report = make_report()
        updated, record = apply_review(
            report, ReviewDecision.APPROVE, reviewer_id="cs-officer-7"
        )
        assert updated.status is ReportStatus.APPROVED
        assert updated.suggested_actions == report.suggested_actions
        assert record.report_id == "RPT-0001"
        assert record.decision is ReviewDecision.APPROVE
        assert record.reviewer_id == "cs-officer-7"
        assert record.resulting_status is ReportStatus.APPROVED
        # original untouched
        assert report.status is ReportStatus.PENDING_REVIEW

    def test_approve_with_overridden_actions_raises(self):
        with pytest.raises(ReviewError, match="override"):
            apply_review(
                make_report(),
                ReviewDecision.APPROVE,
                reviewer_id="cs-1",
                overridden_actions=OVERRIDE_ACTIONS,
            )


class TestOverride:
    def test_override_replaces_actions(self):
        updated, record = apply_review(
            make_report(),
            ReviewDecision.OVERRIDE,
            reviewer_id="cs-1",
            note="Refund not applicable; credit instead.",
            overridden_actions=OVERRIDE_ACTIONS,
        )
        assert updated.status is ReportStatus.OVERRIDDEN
        assert updated.suggested_actions == OVERRIDE_ACTIONS
        assert record.note == "Refund not applicable; credit instead."

    def test_override_requires_actions(self):
        with pytest.raises(ReviewError, match="overridden_actions"):
            apply_review(
                make_report(), ReviewDecision.OVERRIDE, reviewer_id="cs-1"
            )

    def test_override_with_empty_actions_raises(self):
        with pytest.raises(ReviewError, match="overridden_actions"):
            apply_review(
                make_report(),
                ReviewDecision.OVERRIDE,
                reviewer_id="cs-1",
                overridden_actions=[],
            )


class TestReject:
    def test_reject_transitions_to_rejected(self):
        updated, _ = apply_review(
            make_report(), ReviewDecision.REJECT, reviewer_id="cs-1"
        )
        assert updated.status is ReportStatus.REJECTED


class TestInvalidTransitions:
    @pytest.mark.parametrize(
        "status",
        [
            ReportStatus.APPROVED,
            ReportStatus.OVERRIDDEN,
            ReportStatus.REJECTED,
            ReportStatus.PROCESSING_FAILED,
        ],
    )
    def test_only_pending_review_is_reviewable(self, status):
        with pytest.raises(ReviewError, match="pending_review"):
            apply_review(
                make_report(status=status, needs_human_review=True,
                            review_reason="x"),
                ReviewDecision.APPROVE,
                reviewer_id="cs-1",
            )

    def test_blank_reviewer_id_raises(self):
        with pytest.raises(ReviewError, match="reviewer"):
            apply_review(make_report(), ReviewDecision.APPROVE, reviewer_id="  ")

    def test_unknown_decision_string_raises_review_error(self):
        with pytest.raises(ReviewError, match="unknown decision"):
            apply_review(make_report(), "escalate", reviewer_id="cs-1")


class TestRawStringDecisions:
    """Callers may pass plain strings; they must behave like the enum."""

    def test_string_override_replaces_actions(self):
        updated, record = apply_review(
            make_report(),
            "override",
            reviewer_id="cs-1",
            overridden_actions=OVERRIDE_ACTIONS,
        )
        assert updated.status is ReportStatus.OVERRIDDEN
        assert updated.suggested_actions == OVERRIDE_ACTIONS
        assert record.decision is ReviewDecision.OVERRIDE

    def test_string_override_without_actions_still_raises(self):
        with pytest.raises(ReviewError, match="overridden_actions"):
            apply_review(make_report(), "override", reviewer_id="cs-1")
