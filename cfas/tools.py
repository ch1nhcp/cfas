"""Retrieval tools over the mock data sources.

Every tool returns the same envelope (ToolResponse) with four states, each
mapping to a distinct agent behavior:

- success:       data + source_ids populated; IDs are citable in the report
- not_found:     input was valid but no record matches -> record missing
                 context, do NOT retry and do NOT invent data
- invalid_input: bad argument; message says how to fix -> agent may correct
                 the arguments and call again
- tool_error:    the data source itself failed. Local JSON reads are
                 deterministic, so retrying is useless -> processing error

Every tool requires `reason`: the LLM must state why it is calling the tool,
so the trace always carries decision rationale. The tools themselves do not
act on it.
"""

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from cfas.models import FeedbackCategory

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CUSTOMERS_FILE = "customers.json"
POLICIES_FILE = "policies.json"
GUIDELINES_FILE = "cs_guidelines.json"


class ToolStatus(StrEnum):
    SUCCESS = "success"
    NOT_FOUND = "not_found"
    TOOL_ERROR = "tool_error"
    INVALID_INPUT = "invalid_input"


class ToolResponse(BaseModel):
    """Uniform tool envelope. The contract is enforced here so no tool can
    return an ambiguous shape (e.g. a failure that still carries data)."""

    model_config = ConfigDict(frozen=True)

    status: ToolStatus
    data: dict | list[dict] | None
    source_ids: list[str]
    message: str | None

    @model_validator(mode="after")
    def _enforce_envelope_contract(self) -> "ToolResponse":
        if self.status is ToolStatus.SUCCESS:
            if self.data is None:
                raise ValueError("success response must carry data")
        else:
            if self.data is not None or self.source_ids:
                raise ValueError("failure response must not carry data/source_ids")
            if not self.message:
                raise ValueError("failure response must carry a message")
        return self


def _load_records(
    filename: str, data_dir: Path
) -> tuple[list[dict] | None, ToolResponse | None]:
    """Load a data file, returning (records, None) or (None, error_response).

    Error messages name only the file, never the full path: they are fed
    back to the LLM and must not leak filesystem details.
    """
    try:
        with open(data_dir / filename, encoding="utf-8") as f:
            records = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return None, _failure(
            ToolStatus.TOOL_ERROR,
            f"failed to read {filename} ({type(exc).__name__})",
        )
    if not isinstance(records, list):
        return None, _failure(
            ToolStatus.TOOL_ERROR, f"{filename} must contain a JSON array"
        )
    return records, None


def _validate_reason(reason: str) -> ToolResponse | None:
    """Reason exists to guarantee decision rationale in the trace; a blank
    one defeats that, so it is rejected like any other invalid input."""
    if not isinstance(reason, str) or not reason.strip():
        return _failure(
            ToolStatus.INVALID_INPUT,
            "reason must be a non-empty explanation of why you call this tool",
        )
    return None


def _success(data: dict | list[dict], source_ids: list[str]) -> ToolResponse:
    return ToolResponse(
        status=ToolStatus.SUCCESS, data=data, source_ids=source_ids, message=None
    )


def _failure(status: ToolStatus, message: str) -> ToolResponse:
    return ToolResponse(status=status, data=None, source_ids=[], message=message)


def _parse_category(raw: str) -> FeedbackCategory | None:
    try:
        return FeedbackCategory(raw)
    except ValueError:
        return None


def _invalid_category(raw: str) -> ToolResponse:
    valid = ", ".join(sorted(c.value for c in FeedbackCategory))
    return _failure(
        ToolStatus.INVALID_INPUT,
        f"unknown category '{raw}'; valid categories: {valid}",
    )


def get_customer(
    customer_id: str, reason: str, *, data_dir: Path = DATA_DIR
) -> ToolResponse:
    """Look up one customer record (tier, tenure, tickets, orders) by ID."""
    if invalid_reason := _validate_reason(reason):
        return invalid_reason
    wanted = customer_id.strip() if isinstance(customer_id, str) else ""
    if not wanted:
        return _failure(
            ToolStatus.INVALID_INPUT,
            "customer_id must be a non-empty string like 'CUST-001'",
        )
    records, error = _load_records(CUSTOMERS_FILE, data_dir)
    if error:
        return error

    for record in records:
        if record.get("customer_id") == wanted:
            return _success(record, [wanted])
    return _failure(ToolStatus.NOT_FOUND, f"no customer record for '{wanted}'")


def search_policies(
    category: str,
    reason: str,
    query: str | None = None,
    *,
    data_dir: Path = DATA_DIR,
) -> ToolResponse:
    """Find company policies applicable to a feedback category, optionally
    narrowed by a free-text query over policy names and rules."""
    if invalid_reason := _validate_reason(reason):
        return invalid_reason
    parsed = _parse_category(category)
    if parsed is None:
        return _invalid_category(category)
    records, error = _load_records(POLICIES_FILE, data_dir)
    if error:
        return error

    matches = [
        p for p in records if parsed.value in p.get("applicable_categories", [])
    ]
    needle = (query or "").strip().lower()
    if needle:
        matches = [
            p
            for p in matches
            if needle in " ".join([p.get("name", ""), *p.get("rules", [])]).lower()
        ]
    if not matches:
        detail = f" matching query '{needle}'" if needle else ""
        return _failure(
            ToolStatus.NOT_FOUND,
            f"no policy applies to category '{parsed.value}'{detail}",
        )
    return _success(matches, [p["policy_id"] for p in matches])


def get_cs_guidelines(
    category: str, reason: str, *, data_dir: Path = DATA_DIR
) -> ToolResponse:
    """Fetch the CS workflow SOP(s) for a feedback category."""
    if invalid_reason := _validate_reason(reason):
        return invalid_reason
    parsed = _parse_category(category)
    if parsed is None:
        return _invalid_category(category)
    records, error = _load_records(GUIDELINES_FILE, data_dir)
    if error:
        return error

    matches = [g for g in records if g.get("category") == parsed.value]
    if not matches:
        return _failure(
            ToolStatus.NOT_FOUND, f"no CS guideline for category '{parsed.value}'"
        )
    return _success(matches, [g["guideline_id"] for g in matches])


_REASON_PROPERTY = {
    "type": "string",
    "description": "Why you are calling this tool right now; recorded in the audit trace.",
}
_CATEGORY_PROPERTY = {
    "type": "string",
    "enum": [c.value for c in FeedbackCategory],
    "description": "Feedback category from the classification step.",
}

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "get_customer",
        "description": (
            "Look up a customer record (tier, tenure, past tickets, order "
            "history) by customer ID."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "string",
                    "description": "Customer ID, e.g. 'CUST-001'.",
                },
                "reason": _REASON_PROPERTY,
            },
            "required": ["customer_id", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_policies",
        "description": (
            "Find company policies (refund, SLA, escalation, ...) applicable "
            "to a feedback category. Optional free-text query narrows results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": _CATEGORY_PROPERTY,
                "query": {
                    "type": "string",
                    "description": "Optional keywords to narrow the search, e.g. 'refund'.",
                },
                "reason": _REASON_PROPERTY,
            },
            "required": ["category", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_cs_guidelines",
        "description": (
            "Fetch the standard CS workflow (SOP) for handling a feedback "
            "category."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": _CATEGORY_PROPERTY,
                "reason": _REASON_PROPERTY,
            },
            "required": ["category", "reason"],
            "additionalProperties": False,
        },
    },
]

TOOL_HANDLERS = {
    "get_customer": get_customer,
    "search_policies": search_policies,
    "get_cs_guidelines": get_cs_guidelines,
}
