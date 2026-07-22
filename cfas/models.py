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

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


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
    """A raw feedback submission plus minimal metadata. System boundary input."""

    model_config = ConfigDict(frozen=True)

    feedback_text: str = Field(min_length=1)
    customer_id: str | None
    timestamp: datetime
    channel: Channel


class Classification(BaseModel):
    """LLM-facing schema for the classification step."""

    model_config = ConfigDict(frozen=True)

    category: FeedbackCategory
    sentiment: Sentiment
    urgency: Urgency
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    is_ambiguous: bool


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


class ReportDraft(BaseModel):
    """LLM-facing schema for report generation.

    Deliberately contains NO control fields (status, needs_human_review,
    warnings, missing_context): those are computed by the validation gate and
    set by code when assembling the final FeedbackReport.
    """

    model_config = ConfigDict(frozen=True)

    summary: str = Field(min_length=1)
    customer_context: str
    workflow_references: list[str]
    policy_references: list[str]
    suggested_actions: list[SuggestedAction]
    confidence: float = Field(ge=0.0, le=1.0)


class FeedbackReport(BaseModel):
    """Final report assembled by code from a ReportDraft + validation gate
    results. The LLM never produces this model directly."""

    model_config = ConfigDict(frozen=True)

    report_id: str
    summary: str
    customer_context: str
    workflow_references: list[str]
    policy_references: list[str]
    suggested_actions: list[SuggestedAction]
    confidence: float = Field(ge=0.0, le=1.0)
    classification: Classification | None
    warnings: list[str]
    missing_context: list[str]
    needs_human_review: bool
    review_reason: str | None
    status: ReportStatus
