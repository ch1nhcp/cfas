"""Core data contracts for the feedback pipeline.

Design principles:

- Two kinds of schema live here. LLM-facing schemas (Classification, ReportDraft)
  define what the model is *allowed* to produce - they deliberately contain no
  control fields. Code-owned schemas (FeedbackReport) carry the control fields
  (status, needs_human_review, warnings, ...) that only the pipeline may set.
  The LLM cannot set a field that is not in its own schema.

- No defaults on pipeline models. A field the LLM forgot must surface as a
  ValidationError (triggering repair/retry), not be silently filled in.
  Nullable fields are `X | None` without a default: None must be passed
  explicitly.

- All models are frozen. Pipeline stages return new objects instead of
  mutating shared state.
"""

import re
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Mock-data convention; validated at intake so arbitrary strings (or
# prompt-injection payloads) can never ride along as a customer ID.
CUSTOMER_ID_PATTERN = re.compile(r"^CUST-\d{3}$")


class Channel(StrEnum):
    """Where the feedback came from."""

    EMAIL = "email"
    CHAT = "chat"
    WEB_FORM = "web_form"
    PHONE = "phone"
    SOCIAL = "social"


class FeedbackCategory(StrEnum):
    """Feedback taxonomy. Single source of truth: mock data and prompts must
    use these exact values."""

    BUG_REPORT = "bug_report"
    BILLING_COMPLAINT = "billing_complaint"
    FEATURE_REQUEST = "feature_request"
    PRAISE = "praise"
    CHURN_RISK = "churn_risk"
    ABUSE_POLICY_VIOLATION = "abuse_policy_violation"
    OTHER = "other"


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class Urgency(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ActionType(StrEnum):
    """Discriminates suggested actions so the validation gate can check
    citation requirements per type instead of parsing action text."""

    POLICY_ACTION = "policy_action"
    ESCALATION = "escalation"
    CUSTOMER_RESPONSE = "customer_response"
    MANUAL_TRIAGE = "manual_triage"
    LOG_ONLY = "log_only"


class ReportStatus(StrEnum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    OVERRIDDEN = "overridden"
    REJECTED = "rejected"
    PROCESSING_FAILED = "processing_failed"


class FeedbackSubmission(BaseModel):
    """A raw feedback submission plus minimal metadata. System boundary input.

    Normalization lives on the model so the invariants hold no matter which
    entry point constructs it: feedback_text is stripped (whitespace-only
    fails min_length), a blank customer_id becomes None (anonymous).
    """

    model_config = ConfigDict(frozen=True)

    feedback_text: str = Field(min_length=1)
    customer_id: str | None
    timestamp: datetime
    channel: Channel

    @field_validator("feedback_text", mode="before")
    @classmethod
    def _strip_feedback_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("customer_id", mode="before")
    @classmethod
    def _blank_customer_id_to_none(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("customer_id", mode="after")
    @classmethod
    def _customer_id_matches_convention(cls, value: str | None) -> str | None:
        if value is not None and not CUSTOMER_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "customer_id must match 'CUST-NNN' (e.g. CUST-001)"
            )
        return value


class Classification(BaseModel):
    """LLM-facing schema for the classification step."""

    model_config = ConfigDict(frozen=True)

    category: FeedbackCategory
    sentiment: Sentiment
    urgency: Urgency
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    is_ambiguous: bool

    @field_validator("reason", mode="before")
    @classmethod
    def _strip_reason(cls, value: object) -> object:
        # whitespace-only must fail min_length, triggering repair
        return value.strip() if isinstance(value, str) else value


class SuggestedAction(BaseModel):
    """One suggested next step for the CS officer.

    Rule (enforced by the validation gate, not here): source_ids may be empty
    only when action_type is manual_triage or log_only - every other action
    must cite the policy/SOP it is based on.
    """

    model_config = ConfigDict(frozen=True)

    action_type: ActionType
    action: str = Field(min_length=1)
    source_ids: list[str]

    @field_validator("action", mode="before")
    @classmethod
    def _strip_action(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class ReportDraft(BaseModel):
    """LLM-facing schema for report generation.

    Deliberately contains NO control fields (status, needs_human_review,
    warnings, missing_context): those are computed by the validation gate and
    set by code when assembling the final FeedbackReport.
    """

    model_config = ConfigDict(frozen=True)

    summary: str = Field(min_length=1)
    customer_context: str = Field(min_length=1)
    workflow_references: list[str]
    policy_references: list[str]
    # at least one action always - "log_only"/"manual_triage" exist
    # precisely so there is never a reason to return none
    suggested_actions: list[SuggestedAction] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("summary", "customer_context", mode="before")
    @classmethod
    def _strip_text_fields(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class FeedbackReport(ReportDraft):
    """Final report: the draft fields plus code-owned control fields.

    Assembled by code from a ReportDraft + validation gate results; the LLM
    never produces this model directly. classification is None only for
    processing-failure reports where classification itself failed."""

    report_id: str
    classification: Classification | None
    warnings: list[str]
    missing_context: list[str]
    needs_human_review: bool
    review_reason: str | None
    status: ReportStatus
