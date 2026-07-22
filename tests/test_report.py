"""Report generation tests: LLM drafts, code assembles and controls."""

import pytest
from factories import make_classification, make_retrieval
from fakes import FakeClient, make_text_response

from cfas.agent import Source, SourceStatus
from cfas.gate import validate_context
from cfas.models import ReportStatus
from cfas.report import ReportGenerationError, generate_report

VALID_DRAFT_JSON = """{
  "summary": "Customer was double-charged for their subscription.",
  "customer_context": "CUST-001, premium tier, one prior billing ticket.",
  "workflow_references": ["SOP-BILLING-001"],
  "policy_references": ["POL-REFUND-001"],
  "suggested_actions": [
    {
      "action_type": "policy_action",
      "action": "Issue refund per POL-REFUND-001.",
      "source_ids": ["POL-REFUND-001"]
    }
  ],
  "confidence": 0.88
}"""


def run_generate(client, classification=None, retrieval=None):
    classification = classification or make_classification()
    retrieval = retrieval or make_retrieval()
    context_validation = validate_context(classification, retrieval)
    return generate_report(
        classification=classification,
        retrieval=retrieval,
        context_validation=context_validation,
        client=client,
        report_id="RPT-TEST-1",
    )


class TestHappyPath:
    def test_assembles_final_report(self):
        client = FakeClient([make_text_response(VALID_DRAFT_JSON)])
        report = run_generate(client)
        assert report.report_id == "RPT-TEST-1"
        assert report.status is ReportStatus.PENDING_REVIEW
        assert report.needs_human_review is False
        assert report.review_reason is None
        assert report.classification == make_classification()
        assert report.policy_references == ["POL-REFUND-001"]

    def test_status_is_always_code_set(self):
        # even a clean report starts pending_review - the LLM cannot set it
        client = FakeClient([make_text_response(VALID_DRAFT_JSON)])
        report = run_generate(client)
        assert report.status is ReportStatus.PENDING_REVIEW

    def test_request_contains_validated_context_and_ids(self):
        client = FakeClient([make_text_response(VALID_DRAFT_JSON)])
        run_generate(client)
        content = client.requests[0]["messages"][0]["content"]
        assert "POL-REFUND-001" in content  # retrieved ids listed
        assert "billing_complaint" in content  # classification present
        assert client.requests[0]["output_config"]["format"]["type"] == "json_schema"


class TestGateIntegration:
    def test_ungrounded_reference_stripped_and_review_forced(self):
        hallucinated = VALID_DRAFT_JSON.replace(
            '"policy_references": ["POL-REFUND-001"]',
            '"policy_references": ["POL-REFUND-001", "POL-GHOST-777"]',
        )
        client = FakeClient([make_text_response(hallucinated)])
        report = run_generate(client)
        assert report.policy_references == ["POL-REFUND-001"]
        assert report.needs_human_review is True
        assert "POL-GHOST-777" in " ".join(report.warnings)

    def test_context_phase_reasons_propagate(self):
        retrieval = make_retrieval(
            source_states={
                Source.CUSTOMER: SourceStatus.NOT_FOUND,
                Source.POLICIES: SourceStatus.RETRIEVED,
                Source.GUIDELINES: SourceStatus.RETRIEVED,
            },
            missing_context=["customer: no matching record found"],
        )
        client = FakeClient([make_text_response(VALID_DRAFT_JSON)])
        report = run_generate(client, retrieval=retrieval)
        assert report.needs_human_review is True
        assert report.missing_context == ["customer: no matching record found"]
        assert "missing" in report.review_reason

    def test_low_draft_confidence_forces_review(self):
        low = VALID_DRAFT_JSON.replace("0.88", "0.4")
        client = FakeClient([make_text_response(low)])
        report = run_generate(client)
        assert report.needs_human_review is True
        assert report.confidence == 0.4


class TestRepairRetry:
    def test_invalid_draft_repaired_once(self):
        bad = VALID_DRAFT_JSON.replace("policy_action", "do_stuff")
        client = FakeClient(
            [make_text_response(bad), make_text_response(VALID_DRAFT_JSON)]
        )
        report = run_generate(client)
        assert report.status is ReportStatus.PENDING_REVIEW
        assert len(client.requests) == 2

    def test_second_failure_raises(self):
        bad = VALID_DRAFT_JSON.replace("policy_action", "do_stuff")
        client = FakeClient([make_text_response(bad), make_text_response(bad)])
        with pytest.raises(ReportGenerationError):
            run_generate(client)


class TestReportIds:
    def test_generated_report_id_when_not_supplied(self):
        client = FakeClient([make_text_response(VALID_DRAFT_JSON)])
        classification = make_classification()
        retrieval = make_retrieval()
        report = generate_report(
            classification=classification,
            retrieval=retrieval,
            context_validation=validate_context(classification, retrieval),
            client=client,
        )
        assert report.report_id.startswith("RPT-")
        assert len(report.report_id) > 4
