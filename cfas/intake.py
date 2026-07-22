"""Feedback intake: the boundary where free text + metadata become a
validated, immutable FeedbackSubmission.

Normalization rules (strip text, blank customer_id -> None) live on the
model itself; this layer only supplies the receive timestamp.
"""

from datetime import datetime, timezone

from cfas.models import FeedbackSubmission


def build_submission(
    feedback_text: str,
    customer_id: str | None,
    channel: str,
    timestamp: datetime | None = None,
) -> FeedbackSubmission:
    """Validate raw intake values into a FeedbackSubmission.

    Raises pydantic.ValidationError on empty/whitespace-only feedback text
    or unknown channel.
    """
    return FeedbackSubmission(
        feedback_text=feedback_text,
        customer_id=customer_id,
        timestamp=timestamp or datetime.now(timezone.utc),
        channel=channel,
    )
