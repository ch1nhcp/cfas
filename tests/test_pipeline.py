"""End-to-end pipeline tests with a fully scripted LLM."""

from datetime import datetime, timezone

import anthropic
import pytest
from factories import customer_call, guidelines_call, policies_call
from fakes import FakeAuthError, FakeClient, FakeRateLimit, make_response, make_text_response

from cfas.intake import build_submission
from cfas.models import ReportStatus
from cfas.pipeline import process_feedback

SUBMISSION = build_submission(
    feedback_text="I was charged twice for my subscription.",
    customer_id="CUST-001",
    channel="email",
    timestamp=datetime(2026, 7, 20, tzinfo=timezone.utc),
)

CLASSIFY_JSON = """{
  "category": "billing_complaint",
  "sentiment": "negative",
  "urgency": "medium",
  "confidence": 0.92,
  "reason": "Duplicate subscription charge.",
  "is_ambiguous": false
}"""

DRAFT_JSON = """{
  "summary": "Customer was double-charged for their subscription.",
  "customer_context": "CUST-001, premium tier.",
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


def happy_responses():
    return [
        make_text_response(CLASSIFY_JSON),
        make_response(customer_call(), policies_call(), guidelines_call()),
        make_text_response(DRAFT_JSON),
    ]


def run(client):
    return process_feedback(SUBMISSION, client=client, sleep=lambda _s: None)


class TestHappyPath:
    def test_full_pipeline_produces_pending_report(self):
        result = run(FakeClient(happy_responses()))
        assert result.report.status is ReportStatus.PENDING_REVIEW
        assert result.report.needs_human_review is False
        assert result.report.policy_references == ["POL-REFUND-001"]

    def test_trace_records_all_steps_and_rationale(self):
        result = run(FakeClient(happy_responses()))
        steps = [s["step"] for s in result.trace["steps"]]
        assert steps == [
            "intake",
            "classification",
            "retrieval",
            "validate_context",
            "report",
            "final",
        ]
        retrieval_step = result.trace["steps"][2]
        reasons = [call["arguments"]["reason"] for call in retrieval_step["tool_calls"]]
        assert all(reasons)  # decision rationale always present
        assert result.trace["steps"][-1]["retry_count"] == 0
        assert result.trace["run_id"].startswith("RUN-")


class TestFailureFallback:
    def test_classification_hard_failure_yields_failure_report(self):
        bad = CLASSIFY_JSON.replace("billing_complaint", "spam_report")
        result = run(FakeClient([make_text_response(bad), make_text_response(bad)]))
        report = result.report
        assert report.status is ReportStatus.PROCESSING_FAILED
        assert report.needs_human_review is True
        assert report.classification is None
        assert report.suggested_actions[0].action_type.value == "manual_triage"
        assert any(
            s["step"] == "processing_failed" for s in result.trace["steps"]
        )

    def test_report_stage_failure_keeps_classification(self):
        bad = DRAFT_JSON.replace("policy_action", "do_stuff")
        responses = [
            make_text_response(CLASSIFY_JSON),
            make_response(customer_call(), policies_call(), guidelines_call()),
            make_text_response(bad),
            make_text_response(bad),
        ]
        result = run(FakeClient(responses))
        assert result.report.status is ReportStatus.PROCESSING_FAILED
        assert result.report.classification is not None

    def test_llm_dead_after_all_retries_yields_failure_report(self):
        result = run(FakeClient([FakeRateLimit() for _ in range(3)]))
        assert result.report.status is ReportStatus.PROCESSING_FAILED
        assert "rate limited" in result.report.review_reason


class TestEndToEndEdgeCases:
    def test_ambiguous_classification_flows_to_forced_review(self):
        ambiguous = CLASSIFY_JSON.replace('"confidence": 0.92', '"confidence": 0.5')
        result = run(
            FakeClient(
                [
                    make_text_response(ambiguous),
                    make_response(customer_call(), policies_call(), guidelines_call()),
                    make_text_response(DRAFT_JSON),
                ]
            )
        )
        report = result.report
        assert report.status is ReportStatus.PENDING_REVIEW
        assert report.classification.is_ambiguous is True  # forced by code
        assert report.needs_human_review is True
        assert "confidence" in report.review_reason

    def test_missing_customer_record_flows_to_missing_context(self):
        missing = build_submission(
            feedback_text="Where is my refund?",
            customer_id="CUST-404",
            channel="email",
            timestamp=datetime(2026, 7, 20, tzinfo=timezone.utc),
        )
        responses = [
            make_text_response(CLASSIFY_JSON),
            make_response(
                customer_call("CUST-404"), policies_call(), guidelines_call()
            ),
            # final-chance turn after the not_found
            make_text_response("Customer record does not exist; done."),
            make_text_response(DRAFT_JSON.replace("CUST-001, premium tier.",
                                                  "Customer record not found.")),
        ]
        result = process_feedback(missing, client=FakeClient(responses),
                                  sleep=lambda _s: None)
        report = result.report
        assert report.status is ReportStatus.PENDING_REVIEW
        assert any("customer" in note for note in report.missing_context)
        assert report.needs_human_review is True


class TestConfigErrorsCrashLoudly:
    def test_auth_error_is_not_converted_into_failure_report(self):
        with pytest.raises(anthropic.AuthenticationError):
            run(FakeClient([FakeAuthError()]))


class TestTransientRecovery:
    def test_transient_error_recovered_and_counted(self):
        responses = [FakeRateLimit(), *happy_responses()]
        result = run(FakeClient(responses))
        assert result.report.status is ReportStatus.PENDING_REVIEW
        assert result.trace["steps"][-1]["retry_count"] == 1
