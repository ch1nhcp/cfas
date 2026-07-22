"""Classification step tests.

The Anthropic client is faked: tests pin the request contract (model,
structured-output schema, untrusted-content wrapping) and the control logic
(Pydantic validation, one repair retry, deterministic ambiguity rule).
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fakes import FakeClient, make_text_response

from cfas.classify import ClassificationError, classify_feedback
from cfas.config import AMBIGUITY_THRESHOLD, MODEL_ID
from cfas.intake import build_submission
from cfas.llm import structured_output_schema
from cfas.models import Classification


def classification_output_schema():
    return structured_output_schema(Classification)

SUBMISSION = build_submission(
    feedback_text="I was charged twice for my subscription this month.",
    customer_id="CUST-001",
    channel="email",
    timestamp=datetime(2026, 7, 20, tzinfo=timezone.utc),
)

VALID_JSON = """{
  "category": "billing_complaint",
  "sentiment": "negative",
  "urgency": "medium",
  "confidence": 0.92,
  "reason": "Customer reports a duplicate subscription charge.",
  "is_ambiguous": false
}"""


class TestHappyPath:
    def test_returns_validated_classification(self):
        client = FakeClient([make_text_response(VALID_JSON)])
        c = classify_feedback(SUBMISSION, client=client)
        assert c.category.value == "billing_complaint"
        assert c.confidence == 0.92
        assert c.is_ambiguous is False
        assert len(client.requests) == 1

    def test_request_contract(self):
        client = FakeClient([make_text_response(VALID_JSON)])
        classify_feedback(SUBMISSION, client=client)
        req = client.requests[0]
        assert req["model"] == MODEL_ID
        assert req["output_config"]["format"]["type"] == "json_schema"
        # no sampling params: removed on the target model (400 if sent)
        assert "temperature" not in req and "top_p" not in req

    def test_feedback_is_wrapped_as_untrusted_content(self):
        client = FakeClient([make_text_response(VALID_JSON)])
        classify_feedback(SUBMISSION, client=client)
        user_content = client.requests[0]["messages"][0]["content"]
        assert "<customer_feedback>" in user_content
        assert SUBMISSION.feedback_text in user_content
        assert "untrusted" in client.requests[0]["system"].lower()


class TestAmbiguityRule:
    def test_low_confidence_forces_ambiguous(self):
        low = VALID_JSON.replace("0.92", str(AMBIGUITY_THRESHOLD - 0.1))
        client = FakeClient([make_text_response(low)])
        c = classify_feedback(SUBMISSION, client=client)
        assert c.is_ambiguous is True  # code overrides the LLM's false

    def test_llm_ambiguous_flag_is_kept_when_confident(self):
        flagged = VALID_JSON.replace('"is_ambiguous": false', '"is_ambiguous": true')
        client = FakeClient([make_text_response(flagged)])
        c = classify_feedback(SUBMISSION, client=client)
        assert c.is_ambiguous is True  # never downgraded

    def test_confident_unambiguous_stays_unambiguous(self):
        client = FakeClient([make_text_response(VALID_JSON)])
        c = classify_feedback(SUBMISSION, client=client)
        assert c.is_ambiguous is False


class TestRepairRetry:
    def test_invalid_category_repaired_on_second_call(self):
        bad = VALID_JSON.replace("billing_complaint", "spam_report")
        client = FakeClient([make_text_response(bad), make_text_response(VALID_JSON)])
        c = classify_feedback(SUBMISSION, client=client)
        assert c.category.value == "billing_complaint"
        assert len(client.requests) == 2

    def test_repair_request_carries_validation_errors(self):
        bad = VALID_JSON.replace("0.92", "1.7")  # out of [0,1]
        client = FakeClient([make_text_response(bad), make_text_response(VALID_JSON)])
        classify_feedback(SUBMISSION, client=client)
        repair_messages = client.requests[1]["messages"]
        assert len(repair_messages) == 3  # user, assistant echo, repair user
        assert repair_messages[1]["role"] == "assistant"
        assert "confidence" in repair_messages[2]["content"]

    def test_second_failure_raises(self):
        bad = VALID_JSON.replace("billing_complaint", "spam_report")
        client = FakeClient([make_text_response(bad), make_text_response(bad)])
        with pytest.raises(ClassificationError):
            classify_feedback(SUBMISSION, client=client)
        assert len(client.requests) == 2  # exactly one repair, no loop


class TestResponseEdgeCases:
    def test_refusal_raises_without_retry(self):
        client = FakeClient([make_text_response("", stop_reason="refusal")])
        with pytest.raises(ClassificationError, match="refus"):
            classify_feedback(SUBMISSION, client=client)
        assert len(client.requests) == 1

    def test_truncated_output_raises(self):
        client = FakeClient([make_text_response("{", stop_reason="max_tokens")])
        with pytest.raises(ClassificationError, match="max_tokens"):
            classify_feedback(SUBMISSION, client=client)

    def test_missing_text_block_raises(self):
        response = SimpleNamespace(content=[], stop_reason="end_turn")
        client = FakeClient([response])
        with pytest.raises(ClassificationError):
            classify_feedback(SUBMISSION, client=client)

    def test_refusal_on_repair_response_raises(self):
        bad = VALID_JSON.replace("billing_complaint", "spam_report")
        client = FakeClient(
            [make_text_response(bad), make_text_response("", stop_reason="refusal")]
        )
        with pytest.raises(ClassificationError, match="refus"):
            classify_feedback(SUBMISSION, client=client)
        assert len(client.requests) == 2  # repair attempted, then hard stop


class TestOutputSchema:
    """The API rejects unsupported JSON Schema constraints; bounds move to
    client-side Pydantic validation."""

    def collect_keys(self, node, found):
        if isinstance(node, dict):
            found.update(node.keys())
            for value in node.values():
                self.collect_keys(value, found)
        elif isinstance(node, list):
            for item in node:
                self.collect_keys(item, found)

    def test_no_unsupported_constraint_keys(self):
        keys = set()
        self.collect_keys(classification_output_schema(), keys)
        assert not keys & {"minimum", "maximum", "minLength", "maxLength"}

    def test_objects_forbid_additional_properties(self):
        schema = classification_output_schema()
        assert schema["additionalProperties"] is False

    def test_all_fields_required(self):
        schema = classification_output_schema()
        assert set(schema["required"]) == set(schema["properties"])
