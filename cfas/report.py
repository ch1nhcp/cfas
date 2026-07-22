"""Report generation: final LLM call -> ReportDraft -> code-assembled
FeedbackReport.

The LLM fills the ReportDraft schema only (no control fields exist in its
schema, so it cannot set status/needs_human_review/warnings). Code then:
1. runs the phase-2 gate (grounding assertion) on the draft,
2. assembles the FeedbackReport from the sanitized draft + both gate
   phases, setting status=pending_review and needs_human_review itself.
"""

import json
import uuid

import anthropic

from cfas.agent import RetrievalResult
from cfas.config import MODEL_ID, REPORT_MAX_TOKENS
from cfas.gate import ContextValidation, ReportValidation, validate_report
from cfas.llm import default_client, structured_llm_call, structured_output_schema
from cfas.models import Classification, FeedbackReport, ReportDraft, ReportStatus


class ReportGenerationError(Exception):
    """The model could not produce a schema-valid ReportDraft."""


REPORT_SYSTEM_PROMPT = """\
You write the final triage report for a customer support (CS) officer,
based ONLY on the classified feedback and the retrieved context provided.

Rules:
- Ground everything in the provided context. Cite ONLY source IDs that
  appear in the retrieved-source-IDs list; never invent or extrapolate IDs.
- workflow_references: cited SOP IDs; policy_references: cited policy IDs.
- suggested_actions: concrete next steps for the CS officer. Each action
  must cite the policy/SOP IDs it is based on in source_ids and name them
  in the text (e.g. "Issue refund per POL-REFUND-001"). Use action types:
  policy_action, escalation, customer_response, manual_triage, log_only.
  Only manual_triage and log_only may have empty source_ids - use them when
  no grounded action is possible.
- customer_context: summarize the retrieved customer record; if it was not
  retrieved, state that plainly.
- confidence: 0-1 for the report as a whole. Lower it when context is
  missing or contradictory - a well-classified feedback with poor retrieval
  still yields a low-confidence report.
- Note missing or failed context honestly in the summary; never fill gaps
  with plausible-sounding data."""


def _render_context(
    classification: Classification,
    retrieval: RetrievalResult,
    context_validation: ContextValidation,
) -> str:
    sections = [
        "Write the triage report for this feedback.",
        f"Classification:\n{classification.model_dump_json(indent=2)}",
        "Retrieved context by source:\n"
        + json.dumps(
            {source.value: data for source, data in retrieval.context.items()},
            indent=2,
        ),
        "Retrieved source IDs (the ONLY IDs you may cite): "
        + (", ".join(retrieval.retrieved_source_ids) or "(none)"),
    ]
    if context_validation.missing_context:
        sections.append(
            "Missing context:\n- " + "\n- ".join(context_validation.missing_context)
        )
    if context_validation.warnings:
        sections.append(
            "Retrieval warnings:\n- " + "\n- ".join(context_validation.warnings)
        )
    return "\n\n".join(sections)


def _base_request(
    classification: Classification,
    retrieval: RetrievalResult,
    context_validation: ContextValidation,
) -> dict:
    return {
        "model": MODEL_ID,
        "max_tokens": REPORT_MAX_TOKENS,
        "system": REPORT_SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": _render_context(
                    classification, retrieval, context_validation
                ),
            }
        ],
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": structured_output_schema(ReportDraft),
            }
        },
    }


def _assemble(
    report_id: str,
    classification: Classification,
    context_validation: ContextValidation,
    report_validation: ReportValidation,
) -> FeedbackReport:
    """Code owns every control field; the LLM contributed only the draft."""
    reasons = [
        *context_validation.review_reasons,
        *report_validation.review_reasons,
    ]
    return FeedbackReport(
        report_id=report_id,
        **report_validation.draft.model_dump(),
        classification=classification,
        warnings=[*context_validation.warnings, *report_validation.warnings],
        missing_context=context_validation.missing_context,
        needs_human_review=bool(reasons),
        review_reason="; ".join(reasons) if reasons else None,
        status=ReportStatus.PENDING_REVIEW,
    )


def generate_report(
    classification: Classification,
    retrieval: RetrievalResult,
    context_validation: ContextValidation,
    client: anthropic.Anthropic | None = None,
    report_id: str | None = None,
) -> FeedbackReport:
    """Generate and validate the final report.

    Raises ReportGenerationError when the model cannot produce a
    schema-valid draft within one repair retry. Transient API errors
    propagate; the pipeline retry policy owns them.
    """
    client = client or default_client()
    draft = structured_llm_call(
        client,
        _base_request(classification, retrieval, context_validation),
        ReportDraft,
        ReportGenerationError,
    )
    report_validation = validate_report(draft, retrieval.retrieved_source_ids)
    return _assemble(
        report_id=report_id or f"RPT-{uuid.uuid4().hex[:10]}",
        classification=classification,
        context_validation=context_validation,
        report_validation=report_validation,
    )
