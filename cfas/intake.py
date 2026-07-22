"""Feedback intake: the boundary where free text + metadata become a
validated, immutable FeedbackSubmission."""

from datetime import datetime, timezone

from pydantic import ValidationError

from cfas.models import FeedbackSubmission


def build_submission(
    feedback_text: str,
    customer_id: str | None,
    channel: str,
    timestamp: datetime | None = None,
) -> FeedbackSubmission:
    """Validate raw intake values into a FeedbackSubmission.

    Raises pydantic.ValidationError on empty feedback text or unknown channel.
    A blank customer_id is normalized to None (anonymous submission).
    """
    text = feedback_text.strip()
    if not text:
        raise ValidationError.from_exception_data(
            "FeedbackSubmission",
            [
                {
                    "type": "string_too_short",
                    "loc": ("feedback_text",),
                    "input": feedback_text,
                    "ctx": {"min_length": 1},
                }
            ],
        )

    normalized_customer_id = (customer_id or "").strip() or None

    return FeedbackSubmission(
        feedback_text=text,
        customer_id=normalized_customer_id,
        timestamp=timestamp or datetime.now(timezone.utc),
        channel=channel,
    )
