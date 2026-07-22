"""Intake is the system boundary: everything past it is validated, typed data."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from cfas.intake import build_submission
from cfas.models import Channel


class TestBuildSubmission:
    def test_builds_valid_submission(self):
        s = build_submission(
            feedback_text="The app crashes when I open settings.",
            customer_id="CUST-001",
            channel="email",
        )
        assert s.feedback_text == "The app crashes when I open settings."
        assert s.customer_id == "CUST-001"
        assert s.channel is Channel.EMAIL
        assert s.timestamp.tzinfo is not None  # timezone-aware

    def test_customer_id_is_optional(self):
        s = build_submission(
            feedback_text="Love the product!",
            customer_id=None,
            channel="web_form",
        )
        assert s.customer_id is None

    def test_rejects_empty_feedback_text(self):
        with pytest.raises(ValidationError):
            build_submission(feedback_text="", customer_id=None, channel="email")

    def test_rejects_whitespace_only_feedback_text(self):
        with pytest.raises(ValidationError):
            build_submission(feedback_text="   \n  ", customer_id=None, channel="email")

    def test_rejects_unknown_channel(self):
        with pytest.raises(ValidationError):
            build_submission(
                feedback_text="Hello", customer_id=None, channel="carrier_pigeon"
            )

    def test_blank_customer_id_normalized_to_none(self):
        s = build_submission(feedback_text="Hi", customer_id="  ", channel="chat")
        assert s.customer_id is None

    def test_explicit_timestamp_is_preserved(self):
        ts = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
        s = build_submission(
            feedback_text="Hi", customer_id=None, channel="chat", timestamp=ts
        )
        assert s.timestamp == ts

    def test_submission_is_frozen(self):
        s = build_submission(feedback_text="Hi", customer_id=None, channel="chat")
        with pytest.raises(ValidationError):
            s.feedback_text = "changed"
