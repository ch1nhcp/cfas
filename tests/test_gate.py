"""Validation gate tests - both phases are pure functions, no LLM."""

from factories import (
    RETRIEVED_IDS,
    make_action,
    make_classification,
    make_draft,
    make_retrieval,
    make_submission,
)

from cfas.config import REPORT_CONFIDENCE_THRESHOLD
from cfas.gate import grounded_id_set, validate_context, validate_report
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

    def test_wrong_type_reference_is_stripped(self):
        # a grounded POL id still does not belong in workflow_references
        draft = make_draft(
            workflow_references=["SOP-BILLING-001", "POL-REFUND-001"]
        )
        v = validate_report(draft, RETRIEVED_IDS)
        assert v.draft.workflow_references == ["SOP-BILLING-001"]
        assert any("wrong type" in w for w in v.warnings)
        assert any("wrong type" in r for r in v.review_reasons)

    def test_action_citing_only_customer_record_forces_review(self):
        draft = make_draft(
            suggested_actions=[make_action(source_ids=["CUST-001"])]
        )
        v = validate_report(draft, RETRIEVED_IDS)
        assert v.draft.suggested_actions[0].source_ids == ["CUST-001"]  # kept
        assert any("policy/SOP" in r for r in v.review_reasons)

    def test_ungrounded_id_in_free_text_is_flagged(self):
        draft = make_draft(
            summary="Refund per POL-REFUND-999 as discussed."
        )
        v = validate_report(draft, RETRIEVED_IDS)
        assert any("POL-REFUND-999" in w for w in v.warnings)
        assert v.review_reasons

    def test_ungrounded_id_in_prose_is_marked_unverified(self):
        draft = make_draft(
            summary="Refund per POL-REFUND-999 as discussed.",
            customer_context="See CUST-777 history.",
            suggested_actions=[
                make_action(action="Apply POL-GHOST-001 and POL-REFUND-001.")
            ],
        )
        v = validate_report(draft, RETRIEVED_IDS)
        assert "[unverified: POL-REFUND-999]" in v.draft.summary
        assert "[unverified: CUST-777]" in v.draft.customer_context
        action_text = v.draft.suggested_actions[0].action
        assert "[unverified: POL-GHOST-001]" in action_text
        assert "[unverified: POL-REFUND-001]" not in action_text  # grounded
        # original draft untouched
        assert "POL-REFUND-999" in draft.summary
        assert "[unverified" not in draft.summary

    def test_multi_segment_hallucinated_id_is_caught(self):
        draft = make_draft(summary="Apply POL-REFUND-V2-001 here.")
        v = validate_report(draft, RETRIEVED_IDS)
        assert any("POL-REFUND-V2-001" in w for w in v.warnings)

    def test_grounded_id_in_free_text_is_fine(self):
        draft = make_draft(summary="Refund per POL-REFUND-001.")
        v = validate_report(draft, RETRIEVED_IDS)
        assert v.review_reasons == []


class TestGroundedIdSet:
    def test_includes_retrieved_ids(self):
        assert set(RETRIEVED_IDS) <= set(grounded_id_set(make_retrieval()))

    def test_includes_ids_quoted_inside_retrieved_data(self):
        retrieval = make_retrieval(
            context={
                Source.CUSTOMER: [],
                Source.POLICIES: [],
                Source.GUIDELINES: [
                    [{"required_steps": ["Apply POL-ESCALATION-001 rules."]}]
                ],
            }
        )
        assert "POL-ESCALATION-001" in grounded_id_set(retrieval)

    def test_includes_submission_customer_id(self):
        submission = make_submission(customer_id="CUST-404")
        assert "CUST-404" in grounded_id_set(make_retrieval(), submission)

    def test_honest_mention_of_missing_customer_is_not_flagged(self):
        # the CUST-404 demo path: the report honestly names the ID it
        # could not find - that is not a hallucination
        submission = make_submission(customer_id="CUST-404")
        draft = make_draft(
            customer_context="No customer record found for CUST-404."
        )
        v = validate_report(
            draft,
            RETRIEVED_IDS,
            prose_grounded_ids=grounded_id_set(make_retrieval(), submission),
        )
        assert v.review_reasons == []
        assert v.warnings == []


class TestStrictVsProseGrounding:
    """Structured citations require direct retrieval; prose may quote IDs
    that merely appear inside retrieved content."""

    PROSE_IDS = RETRIEVED_IDS + ["POL-ESCALATION-001"]  # quoted in SOP text

    def test_prose_grounded_id_cannot_be_cited_structurally(self):
        draft = make_draft(
            policy_references=["POL-REFUND-001", "POL-ESCALATION-001"],
            summary="Escalate per POL-ESCALATION-001 if unresolved.",
        )
        v = validate_report(draft, RETRIEVED_IDS, prose_grounded_ids=self.PROSE_IDS)
        assert v.draft.policy_references == ["POL-REFUND-001"]  # stripped
        assert any("POL-ESCALATION-001" in w for w in v.warnings)
        assert "[unverified" not in v.draft.summary  # prose mention is fine

    def test_action_citing_prose_only_id_forces_review(self):
        draft = make_draft(
            suggested_actions=[make_action(source_ids=["POL-ESCALATION-001"])]
        )
        v = validate_report(draft, RETRIEVED_IDS, prose_grounded_ids=self.PROSE_IDS)
        assert v.draft.suggested_actions[0].source_ids == []
        assert any("policy/SOP" in r for r in v.review_reasons)


class TestCaseInsensitiveScan:
    def test_lowercase_hallucinated_id_is_caught(self):
        draft = make_draft(summary="Refund per pol-refund-999.")
        v = validate_report(draft, RETRIEVED_IDS)
        assert any("pol-refund-999" in w for w in v.warnings)

    def test_lowercase_mention_of_grounded_id_is_fine(self):
        draft = make_draft(summary="Refund per pol-refund-001.")
        v = validate_report(draft, RETRIEVED_IDS)
        assert v.review_reasons == []


class TestTicketOrderIds:
    def test_fabricated_ticket_and_order_ids_are_caught(self):
        draft = make_draft(
            customer_context="Past ticket TCK-9999 and order ORD-FAKE-999."
        )
        v = validate_report(draft, RETRIEVED_IDS)
        assert any("TCK-9999" in w for w in v.warnings)
        assert "[unverified: TCK-9999]" in v.draft.customer_context
        assert "[unverified: ORD-FAKE-999]" in v.draft.customer_context

    def test_ticket_id_from_retrieved_record_is_grounded(self):
        retrieval = make_retrieval(
            context={
                Source.CUSTOMER: [
                    {"past_tickets": [{"ticket_id": "TCK-1041"}],
                     "orders": [{"order_id": "ORD-9001"}]}
                ],
                Source.POLICIES: [],
                Source.GUIDELINES: [],
            }
        )
        prose_ids = grounded_id_set(retrieval)
        assert "TCK-1041" in prose_ids and "ORD-9001" in prose_ids
        draft = make_draft(
            customer_context="Prior ticket TCK-1041, order ORD-9001 active."
        )
        v = validate_report(draft, RETRIEVED_IDS, prose_grounded_ids=prose_ids)
        assert v.warnings == []
        assert "[unverified" not in v.draft.customer_context


class TestClassificationReasonGrounding:
    def test_hallucinated_id_in_reason_is_flagged_and_redacted(self):
        c = make_classification(
            reason="Customer cites POL-FAKE-999 which mandates refunds."
        )
        v = validate_report(make_draft(), RETRIEVED_IDS, classification=c)
        assert any("classification reason" in w for w in v.warnings)
        assert any("classification reason" in r for r in v.review_reasons)
        assert "[unverified: POL-FAKE-999]" in v.classification.reason
        assert "POL-FAKE-999" in c.reason  # original untouched
        assert "[unverified" not in c.reason

    def test_clean_reason_passes_through_unchanged(self):
        c = make_classification()
        v = validate_report(make_draft(), RETRIEVED_IDS, classification=c)
        assert v.classification == c
        assert v.review_reasons == []

    def test_without_classification_field_is_none(self):
        v = validate_report(make_draft(), RETRIEVED_IDS)
        assert v.classification is None


class TestValidateReportConfidence:
    def test_low_report_confidence_forces_review(self):
        draft = make_draft(confidence=REPORT_CONFIDENCE_THRESHOLD - 0.05)
        v = validate_report(draft, RETRIEVED_IDS)
        assert any("confidence" in r for r in v.review_reasons)

    def test_threshold_confidence_passes(self):
        draft = make_draft(confidence=REPORT_CONFIDENCE_THRESHOLD)
        v = validate_report(draft, RETRIEVED_IDS)
        assert v.review_reasons == []
