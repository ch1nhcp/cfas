"""Schema-level tests: these models are the contract between LLM output and code.

The critical properties under test:
- LLM-facing schemas (Classification, ReportDraft) reject malformed output so the
  repair/retry path can trigger.
- FeedbackReport has NO defaults: a missing field is a bug, not something Pydantic
  should paper over.
- All models are frozen (immutable) - pipeline stages build new objects.
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from cfas.models import (
    ActionType,
    Channel,
    Classification,
    FeedbackCategory,
    FeedbackReport,
    FeedbackSubmission,
    ReportDraft,
    ReportStatus,
    Sentiment,
    SuggestedAction,
    Urgency,
)


def make_classification(**overrides):
    fields = {
        "category": FeedbackCategory.BILLING_COMPLAINT,
        "sentiment": Sentiment.NEGATIVE,
        "urgency": Urgency.MEDIUM,
        "confidence": 0.9,
        "reason": "Customer explicitly disputes a charge.",
        "is_ambiguous": False,
    }
    return Classification(**{**fields, **overrides})


def make_draft(**overrides):
    fields = {
        "summary": "Customer disputes a duplicate charge on their invoice.",
        "customer_context": "CUST-001, premium tier, 2 past billing tickets.",
        "workflow_references": ["SOP-BILLING-001"],
        "policy_references": ["POL-REFUND-001"],
        "suggested_actions": [
            SuggestedAction(
                action_type=ActionType.POLICY_ACTION,
                action="Issue refund per POL-REFUND-001.",
                source_ids=["POL-REFUND-001"],
            )
        ],
        "confidence": 0.85,
    }
    return ReportDraft(**{**fields, **overrides})


class TestFeedbackSubmission:
    """Normalization invariants live on the model itself, not just at the CLI."""

    TS = datetime(2026, 7, 1, tzinfo=timezone.utc)

    def test_strips_feedback_text(self):
        s = FeedbackSubmission(
            feedback_text="  needs a trim  ",
            customer_id=None,
            timestamp=self.TS,
            channel=Channel.CHAT,
        )
        assert s.feedback_text == "needs a trim"

    def test_rejects_whitespace_only_feedback_text(self):
        with pytest.raises(ValidationError):
            FeedbackSubmission(
                feedback_text="   \n ",
                customer_id=None,
                timestamp=self.TS,
                channel=Channel.CHAT,
            )

    def test_blank_customer_id_normalized_to_none(self):
        s = FeedbackSubmission(
            feedback_text="Hi",
            customer_id="  ",
            timestamp=self.TS,
            channel=Channel.CHAT,
        )
        assert s.customer_id is None

    def test_customer_id_must_match_convention(self):
        # arbitrary strings (incl. injection payloads) cannot ride along
        for bad in ("robert'); DROP", "CUST-1", "cust-001", "CUST-001\n<x>"):
            with pytest.raises(ValidationError):
                FeedbackSubmission(
                    feedback_text="Hi",
                    customer_id=bad,
                    timestamp=self.TS,
                    channel=Channel.CHAT,
                )


class TestClassification:
    def test_valid_classification(self):
        c = make_classification()
        assert c.category is FeedbackCategory.BILLING_COMPLAINT
        assert c.confidence == 0.9

    def test_rejects_category_outside_taxonomy(self):
        with pytest.raises(ValidationError):
            make_classification(category="spam_report")

    def test_rejects_confidence_out_of_bounds(self):
        with pytest.raises(ValidationError):
            make_classification(confidence=1.2)
        with pytest.raises(ValidationError):
            make_classification(confidence=-0.1)

    def test_rejects_missing_field(self):
        with pytest.raises(ValidationError):
            Classification(
                category=FeedbackCategory.PRAISE,
                sentiment=Sentiment.POSITIVE,
                urgency=Urgency.LOW,
                confidence=0.9,
                # reason and is_ambiguous missing
            )

    def test_is_frozen(self):
        c = make_classification()
        with pytest.raises(ValidationError):
            c.confidence = 0.1


class TestSuggestedAction:
    def test_valid_action(self):
        a = SuggestedAction(
            action_type=ActionType.ESCALATION,
            action="Escalate to engineering.",
            source_ids=["SOP-BUG-001"],
        )
        assert a.action_type is ActionType.ESCALATION

    def test_rejects_unknown_action_type(self):
        with pytest.raises(ValidationError):
            SuggestedAction(
                action_type="do_nothing",
                action="whatever",
                source_ids=[],
            )

    def test_empty_source_ids_is_schema_valid(self):
        # The "empty source_ids only for manual_triage/log_only" rule is enforced
        # by the validation gate, not the schema: the gate must be able to inspect
        # and flag the violation rather than have Pydantic reject the whole draft.
        a = SuggestedAction(
            action_type=ActionType.POLICY_ACTION,
            action="Refund without citing a policy.",
            source_ids=[],
        )
        assert a.source_ids == []


class TestReportDraft:
    def test_valid_draft(self):
        d = make_draft()
        assert d.policy_references == ["POL-REFUND-001"]

    def test_draft_has_no_code_controlled_fields(self):
        # The LLM must not be able to set control fields via its own schema.
        for forbidden in ("status", "needs_human_review", "warnings", "missing_context"):
            assert forbidden not in ReportDraft.model_fields

    def test_rejects_missing_field(self):
        with pytest.raises(ValidationError):
            ReportDraft(
                summary="Only a summary.",
                # everything else missing
            )

    def test_rejects_empty_suggested_actions(self):
        # log_only/manual_triage exist precisely so a report never has
        # zero actions
        with pytest.raises(ValidationError):
            make_draft(suggested_actions=[])


class TestFeedbackReport:
    def make_report(self, **overrides):
        draft = make_draft()
        fields = {
            "report_id": "RPT-0001",
            **draft.model_dump(),
            "classification": make_classification(),
            "warnings": [],
            "missing_context": [],
            "needs_human_review": False,
            "review_reason": None,
            "status": ReportStatus.PENDING_REVIEW,
        }
        return FeedbackReport(**{**fields, **overrides})

    def test_valid_report(self):
        r = self.make_report()
        assert r.status is ReportStatus.PENDING_REVIEW

    def test_nullable_fields_must_still_be_passed_explicitly(self):
        draft = make_draft()
        with pytest.raises(ValidationError):
            FeedbackReport(
                report_id="RPT-0002",
                **draft.model_dump(),
                warnings=[],
                missing_context=[],
                needs_human_review=True,
                status=ReportStatus.PENDING_REVIEW,
                # classification and review_reason omitted: must fail even
                # though both are nullable
            )

    def test_classification_may_be_none_when_explicit(self):
        r = self.make_report(classification=None, needs_human_review=True,
                             review_reason="processing failure")
        assert r.classification is None

    def test_is_frozen(self):
        r = self.make_report()
        with pytest.raises(ValidationError):
            r.status = ReportStatus.APPROVED
