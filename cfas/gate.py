"""Two-phase deterministic validation gate. No LLM involved.

Phase 1 - validate_context(): after retrieval, before report generation.
Computes every human-review rule that is decidable from classification +
retrieval alone.

Phase 2 - validate_report(): on the LLM-generated draft. The grounding
assertion lives here: every reference ID in the report - structured fields,
per-action source_ids, and IDs mentioned in free text - must be a subset of
what was actually retrieved. Violations are stripped, warned about, and
force human review. Reports are sanitized, never silently trusted.

The report assembly step combines both phases to set the final
needs_human_review; the LLM never controls any of these fields.
"""

import json
import re

from pydantic import BaseModel, ConfigDict

from cfas.agent import RetrievalResult
from cfas.config import AMBIGUITY_THRESHOLD, REPORT_CONFIDENCE_THRESHOLD
from cfas.models import (
    ActionType,
    Classification,
    FeedbackCategory,
    FeedbackSubmission,
    ReportDraft,
    SuggestedAction,
    Urgency,
)

# ID shapes that must be groundable in retrieved data when they appear
# anywhere in a report. Deliberately broader than the real data conventions
# (any CUST-/POL-/SOP- segment chain) so hallucinated variants like
# POL-REFUND-V2-001 are caught too; case-insensitive so a lowercase mention
# cannot evade the scan. TCK-/ORD- IDs are excluded: they live inside
# retrieved customer records, not in retrieved_source_ids, so scanning them
# would flag legitimately grounded mentions.
_SOURCE_ID_PATTERN = re.compile(
    r"\b(?:CUST|POL|SOP)(?:-[A-Z0-9]+)+\b", re.IGNORECASE
)

# Action types that may legitimately carry no source citations.
_CITATION_EXEMPT_TYPES = frozenset({ActionType.MANUAL_TRIAGE, ActionType.LOG_ONLY})


class ContextValidation(BaseModel):
    """Phase-1 result."""

    model_config = ConfigDict(frozen=True)

    missing_context: list[str]
    warnings: list[str]
    review_reasons: list[str]

    @property
    def needs_human_review(self) -> bool:
        return bool(self.review_reasons)


class ReportValidation(BaseModel):
    """Phase-2 result. draft is the sanitized copy (ungrounded IDs stripped)."""

    model_config = ConfigDict(frozen=True)

    draft: ReportDraft
    warnings: list[str]
    review_reasons: list[str]


def validate_context(
    classification: Classification, retrieval: RetrievalResult
) -> ContextValidation:
    """Phase 1: review rules decidable before the report exists."""
    reasons = []
    if classification.confidence < AMBIGUITY_THRESHOLD:
        reasons.append(
            f"classification confidence {classification.confidence:.2f} "
            f"below {AMBIGUITY_THRESHOLD}"
        )
    elif classification.is_ambiguous:
        reasons.append("classification flagged as ambiguous")
    if classification.urgency is Urgency.HIGH:
        reasons.append("high urgency")
    if classification.category is FeedbackCategory.ABUSE_POLICY_VIOLATION:
        reasons.append("abuse/policy violation category")
    if retrieval.missing_context:
        reasons.append(
            "missing context: " + "; ".join(retrieval.missing_context)
        )
    if retrieval.warnings:
        reasons.append(
            "retrieval anomalies: " + "; ".join(retrieval.warnings)
        )
    return ContextValidation(
        missing_context=retrieval.missing_context,
        warnings=retrieval.warnings,
        review_reasons=list(dict.fromkeys(reasons)),
    )


def grounded_id_set(
    retrieval: RetrievalResult,
    submission: FeedbackSubmission | None = None,
) -> list[str]:
    """Every ID a report may legitimately mention:

    - IDs of retrieved records (retrieved_source_ids),
    - IDs appearing INSIDE retrieved data (e.g. a policy ID quoted by an
      SOP's required_steps) - quoting retrieved content is not hallucination,
    - the submission's own customer ID - a not-found lookup may honestly be
      named in the report ("no record found for CUST-404").
    """
    known = list(retrieval.retrieved_source_ids)
    context_json = json.dumps(
        {source.value: data for source, data in retrieval.context.items()}
    )
    for mention in _SOURCE_ID_PATTERN.findall(context_json):
        if mention not in known:
            known.append(mention)
    if submission and submission.customer_id and submission.customer_id not in known:
        known.append(submission.customer_id)
    return known


def _split_grounded(ids: list[str], known: set[str]) -> tuple[list[str], list[str]]:
    grounded = [i for i in ids if i in known]
    stripped = [i for i in ids if i not in known]
    return grounded, stripped


def _sanitize_action(
    action: SuggestedAction, known: set[str]
) -> tuple[SuggestedAction, list[str], list[str]]:
    """Returns (sanitized action, warnings, review reasons)."""
    grounded, stripped = _split_grounded(action.source_ids, known)
    warnings = [
        f"ungrounded source ID '{i}' stripped from action "
        f"'{action.action_type.value}'"
        for i in stripped
    ]
    reasons = []
    if stripped:
        reasons.append("suggested action cited ungrounded source IDs")
    if not grounded and action.action_type not in _CITATION_EXEMPT_TYPES:
        reasons.append(
            f"action '{action.action_type.value}' lacks grounded source "
            "citations (only manual_triage/log_only may omit them)"
        )
    sanitized = (
        action
        if grounded == action.source_ids
        else action.model_copy(update={"source_ids": grounded})
    )
    return sanitized, warnings, reasons


def _scan_free_text(draft: ReportDraft, known: set[str]) -> list[str]:
    """IDs mentioned in prose must be grounded too - a hallucinated
    'per POL-XXX-999' inside the summary is as bad as one in a list.
    Comparison is case-insensitive (the pattern matches lowercase too)."""
    known_upper = {i.upper() for i in known}
    texts = [
        draft.summary,
        draft.customer_context,
        *[action.action for action in draft.suggested_actions],
    ]
    ungrounded: list[str] = []
    seen_upper: set[str] = set()
    for text in texts:
        for mention in _SOURCE_ID_PATTERN.findall(text):
            upper = mention.upper()
            if upper not in known_upper and upper not in seen_upper:
                seen_upper.add(upper)
                ungrounded.append(mention)
    return ungrounded


def validate_report(
    draft: ReportDraft, retrieved_source_ids: list[str]
) -> ReportValidation:
    """Phase 2: grounding assertion + report confidence rule."""
    known = set(retrieved_source_ids)
    warnings: list[str] = []
    reasons: list[str] = []

    workflow_refs, stripped_wf = _split_grounded(draft.workflow_references, known)
    policy_refs, stripped_pol = _split_grounded(draft.policy_references, known)
    for stripped_id in stripped_wf + stripped_pol:
        warnings.append(f"ungrounded reference '{stripped_id}' stripped")
    if stripped_wf or stripped_pol:
        reasons.append("report cited references not present in retrieved data")

    actions = []
    for action in draft.suggested_actions:
        sanitized, action_warnings, action_reasons = _sanitize_action(action, known)
        actions.append(sanitized)
        warnings.extend(action_warnings)
        reasons.extend(action_reasons)

    for mention in _scan_free_text(draft, known):
        warnings.append(f"ungrounded ID '{mention}' mentioned in report text")
        reasons.append("report text mentions ungrounded source IDs")

    if draft.confidence < REPORT_CONFIDENCE_THRESHOLD:
        reasons.append(
            f"report confidence {draft.confidence:.2f} below "
            f"{REPORT_CONFIDENCE_THRESHOLD}"
        )

    sanitized_draft = draft.model_copy(
        update={
            "workflow_references": workflow_refs,
            "policy_references": policy_refs,
            "suggested_actions": actions,
        }
    )
    return ReportValidation(
        draft=sanitized_draft,
        warnings=warnings,
        review_reasons=list(dict.fromkeys(reasons)),  # dedupe, keep order
    )
