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
    POLICY_ID_PREFIX,
    SOP_ID_PREFIX,
    SOURCE_ID_PREFIXES,
    ActionType,
    Classification,
    FeedbackCategory,
    FeedbackSubmission,
    ReportDraft,
    SuggestedAction,
    Urgency,
)

# ID shapes that must be groundable when they appear anywhere in a report;
# built from the shared prefix conventions in cfas.models. Deliberately
# broader than the real data conventions (any known-prefix segment chain)
# so hallucinated variants like POL-REFUND-V2-001 are caught too;
# case-insensitive so a lowercase mention cannot evade the scan.
# TCK-/ORD- (ticket/order) IDs are included: they live inside retrieved
# customer records, and since the prose grounding set also contains IDs
# quoted inside retrieved content, legitimate mentions (TCK-1041 from a
# retrieved record) ground automatically while fabricated ones (TCK-9999)
# are caught.
_SOURCE_ID_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(prefix.rstrip("-") for prefix in SOURCE_ID_PREFIXES)
    + r")(?:-[A-Z0-9]+)+\b",
    re.IGNORECASE,
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
    has_policy_basis = any(
        i.startswith((POLICY_ID_PREFIX, SOP_ID_PREFIX)) for i in grounded
    )
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


def _check_draft_prose(
    draft: ReportDraft, prose_known_upper: set[str]
) -> tuple[list[str], list[str]]:
    """Warnings/reasons for ungrounded IDs mentioned in the draft's prose."""
    texts = [
        draft.summary,
        draft.customer_context,
        *[action.action for action in draft.suggested_actions],
    ]
    warnings, reasons = [], []
    for mention in _scan_texts(texts, prose_known_upper):
        warnings.append(f"ungrounded ID '{mention}' mentioned in report text")
        reasons.append("report text mentions ungrounded source IDs")
    return warnings, reasons


def _sanitize_classification(
    classification: Classification | None, prose_known_upper: set[str]
) -> tuple[Classification | None, list[str], list[str]]:
    """The classification's free-text reason is embedded verbatim in the
    final report, so it gets the same prose treatment as the draft."""
    if classification is None:
        return None, [], []
    warnings, reasons = [], []
    for mention in _scan_texts([classification.reason], prose_known_upper):
        warnings.append(
            f"ungrounded ID '{mention}' mentioned in classification reason"
        )
        reasons.append("classification reason mentions ungrounded source IDs")
    redacted_reason = _mark_unverified(classification.reason, prose_known_upper)
    if redacted_reason != classification.reason:
        classification = classification.model_copy(
            update={"reason": redacted_reason}
        )
    return classification, warnings, reasons


def _sanitize_draft(
    draft: ReportDraft,
    workflow_refs: list[str],
    policy_refs: list[str],
    actions: list[SuggestedAction],
    prose_known_upper: set[str],
) -> ReportDraft:
    return draft.model_copy(
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


def _sanitize_citations(
    draft: ReportDraft, known: set[str], prose_known_upper: set[str]
) -> tuple[list[str], list[str], list[SuggestedAction], list[str], list[str]]:
    """Strip/type-check all structured citations. Returns
    (workflow_refs, policy_refs, actions, warnings, reasons)."""
    workflow_refs, warnings, reasons = _split_typed_refs(
        draft.workflow_references, known, SOP_ID_PREFIX, "workflow_references"
    )
    policy_refs, pol_warnings, pol_reasons = _split_typed_refs(
        draft.policy_references, known, POLICY_ID_PREFIX, "policy_references"
    )
    warnings.extend(pol_warnings)
    reasons.extend(pol_reasons)
    actions = []
    for action in draft.suggested_actions:
        sanitized, action_warnings, action_reasons = _sanitize_action(
            action, known, prose_known_upper
        )
        actions.append(sanitized)
        warnings.extend(action_warnings)
        reasons.extend(action_reasons)
    return workflow_refs, policy_refs, actions, warnings, reasons


def validate_report(
    draft: ReportDraft,
    retrieved_source_ids: list[str],
    *,
    prose_grounded_ids: list[str] | None = None,
    classification: Classification | None = None,
) -> ReportValidation:
    """Phase 2: grounding assertion + report confidence rule.

    Structured citations (reference lists, per-action source_ids) must be a
    subset of retrieved_source_ids - directly retrieved records only. Prose
    (draft text and the classification reason) is checked against
    prose_grounded_ids (see grounded_id_set; defaults to the strict set).
    Keyword-only: the two ID lists differ only in semantics, so positional
    passing would be swap-prone.
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
    workflow_refs, policy_refs, actions, warnings, reasons = _sanitize_citations(
        draft, known, prose_known_upper
    )

    prose_warnings, prose_reasons = _check_draft_prose(draft, prose_known_upper)
    warnings.extend(prose_warnings)
    reasons.extend(prose_reasons)

    sanitized_classification, cls_warnings, cls_reasons = _sanitize_classification(
        classification, prose_known_upper
    )
    warnings.extend(cls_warnings)
    reasons.extend(cls_reasons)

    if draft.confidence < REPORT_CONFIDENCE_THRESHOLD:
        reasons.append(
            f"report confidence {draft.confidence:.2f} below "
            f"{REPORT_CONFIDENCE_THRESHOLD}"
        )

    return ReportValidation(
        draft=_sanitize_draft(
            draft, workflow_refs, policy_refs, actions, prose_known_upper
        ),
        classification=sanitized_classification,
        warnings=list(dict.fromkeys(warnings)),
        review_reasons=list(dict.fromkeys(reasons)),  # dedupe, keep order
    )
