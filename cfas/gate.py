"""Two-phase deterministic validation gate. No LLM involved.

Phase 1 - validate_context(): after retrieval, before report generation.
Computes every human-review rule that is decidable from classification +
retrieval alone.

Phase 2 - validate_report(): on the LLM-generated draft. The grounding
assertion lives here, with two strictness levels: structured citations
(reference lists, per-action source_ids) must be a subset of what was
DIRECTLY retrieved; prose may additionally mention IDs quoted inside
retrieved content or the submission's own customer ID. Violations are
stripped from citations, marked [unverified] in prose, warned about, and
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

# ID shapes that must be groundable when they appear anywhere in a report.
# Deliberately broader than the real data conventions (any known-prefix
# segment chain) so hallucinated variants like POL-REFUND-V2-001 are caught
# too; case-insensitive so a lowercase mention cannot evade the scan.
# TCK-/ORD- (ticket/order) IDs are included: they live inside retrieved
# customer records, and since the prose grounding set also contains IDs
# quoted inside retrieved content, legitimate mentions (TCK-1041 from a
# retrieved record) ground automatically while fabricated ones (TCK-9999)
# are caught.
_SOURCE_ID_PATTERN = re.compile(
    r"\b(?:CUST|POL|SOP|TCK|ORD)(?:-[A-Z0-9]+)+\b", re.IGNORECASE
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
    """Phase-2 result. draft (and classification, when provided) are the
    sanitized copies: ungrounded IDs stripped from citations, marked
    [unverified] in prose."""

    model_config = ConfigDict(frozen=True)

    draft: ReportDraft
    classification: Classification | None
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
    """Every ID a report may legitimately mention IN PROSE:

    - IDs of retrieved records (retrieved_source_ids),
    - IDs appearing INSIDE retrieved data (e.g. a policy ID quoted by an
      SOP's required_steps) - quoting retrieved content is not hallucination,
    - the submission's own customer ID - a not-found lookup may honestly be
      named in the report ("no record found for CUST-404").

    This set is for prose checks only. Structured citations (reference
    lists, action source_ids) are held to the stricter
    retrieved_source_ids: citing a policy whose content was never actually
    fetched is not grounding.
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


def _split_typed_refs(
    ids: list[str], known: set[str], required_prefix: str, list_name: str
) -> tuple[list[str], list[str], list[str]]:
    """Reference lists are typed: workflow_references holds SOPs,
    policy_references holds policies. Returns (kept, warnings, reasons)."""
    kept, warnings, reasons = [], [], []
    for ref in ids:
        if ref not in known:
            warnings.append(f"ungrounded reference '{ref}' stripped")
            reasons.append("report cited references not present in retrieved data")
        elif not ref.startswith(required_prefix):
            warnings.append(
                f"reference '{ref}' stripped from {list_name} "
                f"(wrong type; expected {required_prefix}*)"
            )
            reasons.append("report listed references under the wrong type")
        else:
            kept.append(ref)
    return kept, warnings, reasons


def _mark_unverified(text: str, known_upper: set[str]) -> str:
    """Rewrite any ungrounded ID mentioned in prose as an inline
    '[unverified: ID]' marker - the reviewer still sees what the model
    claimed, but the claim is visibly not backed by retrieved data.

    Not idempotent: applying it twice would nest markers. validate_report
    runs exactly once per draft (a pipeline retry regenerates a fresh
    draft), so a single application is an invariant, not a coincidence."""

    def _sub(match: re.Match) -> str:
        mention = match.group(0)
        if mention.upper() in known_upper:
            return mention
        return f"[unverified: {mention}]"

    return _SOURCE_ID_PATTERN.sub(_sub, text)


def _sanitize_action(
    action: SuggestedAction, known: set[str], prose_known_upper: set[str]
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
    # a customer record alone is not an actionable basis: non-exempt actions
    # must rest on at least one retrieved policy or SOP
    has_policy_basis = any(i.startswith(("POL-", "SOP-")) for i in grounded)
    if not has_policy_basis and action.action_type not in _CITATION_EXEMPT_TYPES:
        reasons.append(
            f"action '{action.action_type.value}' lacks grounded policy/SOP "
            "citations (only manual_triage/log_only may omit them)"
        )
    redacted_text = _mark_unverified(action.action, prose_known_upper)
    if grounded == action.source_ids and redacted_text == action.action:
        return action, warnings, reasons
    sanitized = action.model_copy(
        update={"source_ids": grounded, "action": redacted_text}
    )
    return sanitized, warnings, reasons


def _scan_texts(texts: list[str], known_upper: set[str]) -> list[str]:
    """IDs mentioned in prose must be grounded too - a hallucinated
    'per POL-XXX-999' inside the summary is as bad as one in a list.
    Comparison is case-insensitive (the pattern matches lowercase too)."""
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
    draft: ReportDraft,
    retrieved_source_ids: list[str],
    prose_grounded_ids: list[str] | None = None,
    classification: Classification | None = None,
) -> ReportValidation:
    """Phase 2: grounding assertion + report confidence rule.

    Structured citations (reference lists, per-action source_ids) must be a
    subset of retrieved_source_ids - directly retrieved records only. Prose
    is checked against prose_grounded_ids (see grounded_id_set; defaults to
    the strict set), which additionally allows quoting IDs that appear
    inside retrieved content and the submission's own customer ID.

    When classification is provided, its free-text reason gets the same
    prose treatment - it is embedded verbatim in the final report, so a
    hallucinated ID there is as visible as one in the summary.
    """
    known = set(retrieved_source_ids)
    prose_known_upper = {
        i.upper()
        for i in (
            prose_grounded_ids
            if prose_grounded_ids is not None
            else retrieved_source_ids
        )
    }
    warnings: list[str] = []
    reasons: list[str] = []

    workflow_refs, wf_warnings, wf_reasons = _split_typed_refs(
        draft.workflow_references, known, "SOP-", "workflow_references"
    )
    policy_refs, pol_warnings, pol_reasons = _split_typed_refs(
        draft.policy_references, known, "POL-", "policy_references"
    )
    warnings.extend(wf_warnings + pol_warnings)
    reasons.extend(wf_reasons + pol_reasons)

    actions = []
    for action in draft.suggested_actions:
        sanitized, action_warnings, action_reasons = _sanitize_action(
            action, known, prose_known_upper
        )
        actions.append(sanitized)
        warnings.extend(action_warnings)
        reasons.extend(action_reasons)

    draft_texts = [
        draft.summary,
        draft.customer_context,
        *[action.action for action in draft.suggested_actions],
    ]
    for mention in _scan_texts(draft_texts, prose_known_upper):
        warnings.append(f"ungrounded ID '{mention}' mentioned in report text")
        reasons.append("report text mentions ungrounded source IDs")

    sanitized_classification = classification
    if classification is not None:
        for mention in _scan_texts([classification.reason], prose_known_upper):
            warnings.append(
                f"ungrounded ID '{mention}' mentioned in classification reason"
            )
            reasons.append("classification reason mentions ungrounded source IDs")
        redacted_reason = _mark_unverified(classification.reason, prose_known_upper)
        if redacted_reason != classification.reason:
            sanitized_classification = classification.model_copy(
                update={"reason": redacted_reason}
            )

    if draft.confidence < REPORT_CONFIDENCE_THRESHOLD:
        reasons.append(
            f"report confidence {draft.confidence:.2f} below "
            f"{REPORT_CONFIDENCE_THRESHOLD}"
        )

    sanitized_draft = draft.model_copy(
        update={
            "summary": _mark_unverified(draft.summary, prose_known_upper),
            "customer_context": _mark_unverified(
                draft.customer_context, prose_known_upper
            ),
            "workflow_references": workflow_refs,
            "policy_references": policy_refs,
            "suggested_actions": actions,
        }
    )
    return ReportValidation(
        draft=sanitized_draft,
        classification=sanitized_classification,
        warnings=list(dict.fromkeys(warnings)),
        review_reasons=list(dict.fromkeys(reasons)),  # dedupe, keep order
    )
