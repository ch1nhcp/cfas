"""Shared domain-object factories for tests."""

from fakes import tool_use_block

from cfas.agent import RetrievalResult, Source, SourceStatus
from cfas.models import (
    ActionType,
    Classification,
    FeedbackCategory,
    ReportDraft,
    Sentiment,
    SuggestedAction,
    Urgency,
)

RETRIEVED_IDS = ["CUST-001", "POL-REFUND-001", "POL-SLA-001", "SOP-BILLING-001"]


def make_classification(**overrides):
    fields = {
        "category": FeedbackCategory.BILLING_COMPLAINT,
        "sentiment": Sentiment.NEGATIVE,
        "urgency": Urgency.MEDIUM,
        "confidence": 0.9,
        "reason": "Clear duplicate charge complaint.",
        "is_ambiguous": False,
    }
    return Classification(**{**fields, **overrides})


def make_retrieval(**overrides):
    fields = {
        "source_states": {s: SourceStatus.RETRIEVED for s in Source},
        "context": {s: [{"stub": True}] for s in Source},
        "retrieved_source_ids": RETRIEVED_IDS,
        "missing_context": [],
        "warnings": [],
        "tool_calls": [],
        "iterations": 1,
    }
    return RetrievalResult(**{**fields, **overrides})


def make_action(**overrides):
    fields = {
        "action_type": ActionType.POLICY_ACTION,
        "action": "Issue refund per POL-REFUND-001.",
        "source_ids": ["POL-REFUND-001"],
    }
    return SuggestedAction(**{**fields, **overrides})


def customer_call(customer_id="CUST-001", reason="look up tier and history", cid="t1"):
    return tool_use_block(
        "get_customer", {"customer_id": customer_id, "reason": reason}, cid
    )


def policies_call(category="billing_complaint", reason="find refund policy", cid="t2"):
    return tool_use_block(
        "search_policies", {"category": category, "reason": reason}, cid
    )


def guidelines_call(category="billing_complaint", reason="find the SOP", cid="t3"):
    return tool_use_block(
        "get_cs_guidelines", {"category": category, "reason": reason}, cid
    )


def make_draft(**overrides):
    fields = {
        "summary": "Customer was double-charged for their subscription.",
        "customer_context": "CUST-001, premium tier, prior billing ticket.",
        "workflow_references": ["SOP-BILLING-001"],
        "policy_references": ["POL-REFUND-001"],
        "suggested_actions": [make_action()],
        "confidence": 0.85,
    }
    return ReportDraft(**{**fields, **overrides})
