"""Validation gate tests - both phases are pure functions, no LLM."""

from factories import (
    RETRIEVED_IDS,
    make_action,
    make_classification,
    make_draft,
    make_retrieval,
)

from cfas.config import REPORT_CONFIDENCE_THRESHOLD
from cfas.gate import validate_context, validate_report
from cfas.models import ActionType, FeedbackCategory, Urgency
from cfas.agent import Source, SourceStatus


class TestValidateContext:
    def test_clean_context_needs_no_review(self):
        v = validate_context(make_classification(), make_retrieval())
        assert v.review_reasons == []
        assert v.needs_human_review is False

    def test_low_classification_confidence(self):
        v = validate_context(
            make_classification(confidence=0.5, is_ambiguous=True), make_retrieval()
        )
        assert v.needs_human_review is True
        assert any("confidence" in r for r in v.review_reasons)

    def test_ambiguous_flag_alone_forces_review(self):
        v = validate_context(
            make_classification(is_ambiguous=True), make_retrieval()
        )
        assert v.needs_human_review is True

    def test_high_urgency_forces_review(self):
        v = validate_context(
            make_classification(urgency=Urgency.HIGH), make_retrieval()
        )
        assert any("urgency" in r for r in v.review_reasons)

    def test_abuse_category_forces_review(self):
        v = validate_context(
            make_classification(category=FeedbackCategory.ABUSE_POLICY_VIOLATION),
            make_retrieval(),
        )
        assert any("abuse" in r for r in v.review_reasons)

    def test_missing_context_forces_review(self):
        retrieval = make_retrieval(
            source_states={
                Source.CUSTOMER: SourceStatus.NOT_FOUND,
                Source.POLICIES: SourceStatus.RETRIEVED,
                Source.GUIDELINES: SourceStatus.RETRIEVED,
            },
            missing_context=["customer: no matching record found"],
        )
        v = validate_context(make_classification(), retrieval)
        assert v.needs_human_review is True
        assert v.missing_context == ["customer: no matching record found"]

    def test_retrieval_warnings_force_review(self):
        retrieval = make_retrieval(warnings=["tool iteration limit (6) reached"])
        v = validate_context(make_classification(), retrieval)
        assert v.needs_human_review is True
        assert v.warnings == ["tool iteration limit (6) reached"]


class TestValidateReportGrounding:
    def test_fully_grounded_report_passes_untouched(self):
        v = validate_report(make_draft(), RETRIEVED_IDS)
        assert v.review_reasons == []
        assert v.warnings == []
        assert v.draft == make_draft()

    def test_ungrounded_policy_reference_is_stripped(self):
        draft = make_draft(
            policy_references=["POL-REFUND-001", "POL-IMAGINARY-999"]
        )
        v = validate_report(draft, RETRIEVED_IDS)
        assert v.draft.policy_references == ["POL-REFUND-001"]
        assert any("POL-IMAGINARY-999" in w for w in v.warnings)
        assert v.review_reasons  # forced review
        # original draft untouched (immutability)
        assert draft.policy_references == ["POL-REFUND-001", "POL-IMAGINARY-999"]

    def test_ungrounded_action_source_id_is_stripped(self):
        draft = make_draft(
            suggested_actions=[
                make_action(source_ids=["POL-REFUND-001", "SOP-FAKE-001"])
            ]
        )
        v = validate_report(draft, RETRIEVED_IDS)
        assert v.draft.suggested_actions[0].source_ids == ["POL-REFUND-001"]
        assert v.review_reasons

    def test_action_left_without_citations_forces_review(self):
        draft = make_draft(
            suggested_actions=[make_action(source_ids=["POL-FAKE-001"])]
        )
        v = validate_report(draft, RETRIEVED_IDS)
        assert v.draft.suggested_actions[0].source_ids == []
        assert any("citation" in r for r in v.review_reasons)

    def test_empty_source_ids_allowed_only_for_exempt_action_types(self):
        for action_type in (ActionType.MANUAL_TRIAGE, ActionType.LOG_ONLY):
            draft = make_draft(
                suggested_actions=[
                    make_action(
                        action_type=action_type,
                        action="Route to manual triage.",
                        source_ids=[],
                    )
                ]
            )
            v = validate_report(draft, RETRIEVED_IDS)
            assert v.review_reasons == [], action_type

    def test_empty_source_ids_on_escalation_forces_review(self):
        draft = make_draft(
            suggested_actions=[
                make_action(action_type=ActionType.ESCALATION, source_ids=[])
            ]
        )
        v = validate_report(draft, RETRIEVED_IDS)
        assert any("citation" in r for r in v.review_reasons)

    def test_ungrounded_id_in_free_text_is_flagged(self):
        draft = make_draft(
            summary="Refund per POL-REFUND-999 as discussed."
        )
        v = validate_report(draft, RETRIEVED_IDS)
        assert any("POL-REFUND-999" in w for w in v.warnings)
        assert v.review_reasons

    def test_multi_segment_hallucinated_id_is_caught(self):
        draft = make_draft(summary="Apply POL-REFUND-V2-001 here.")
        v = validate_report(draft, RETRIEVED_IDS)
        assert any("POL-REFUND-V2-001" in w for w in v.warnings)

    def test_grounded_id_in_free_text_is_fine(self):
        draft = make_draft(summary="Refund per POL-REFUND-001.")
        v = validate_report(draft, RETRIEVED_IDS)
        assert v.review_reasons == []


class TestValidateReportConfidence:
    def test_low_report_confidence_forces_review(self):
        draft = make_draft(confidence=REPORT_CONFIDENCE_THRESHOLD - 0.05)
        v = validate_report(draft, RETRIEVED_IDS)
        assert any("confidence" in r for r in v.review_reasons)

    def test_threshold_confidence_passes(self):
        draft = make_draft(confidence=REPORT_CONFIDENCE_THRESHOLD)
        v = validate_report(draft, RETRIEVED_IDS)
        assert v.review_reasons == []
