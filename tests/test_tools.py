"""Retrieval-tool tests: all four envelope states per tool, plus the
envelope contract itself and the LLM-facing tool schemas."""

import json

import pytest
from pydantic import ValidationError

from cfas.models import FeedbackCategory
from cfas.tools import (
    CUSTOMERS_FILE,
    GUIDELINES_FILE,
    POLICIES_FILE,
    TOOL_DEFINITIONS,
    TOOL_HANDLERS,
    ToolResponse,
    ToolStatus,
    get_cs_guidelines,
    get_customer,
    search_policies,
)

REASON = "unit test"


@pytest.fixture
def broken_data_dir(tmp_path):
    """A data dir where every file exists but contains invalid JSON."""
    for name in (CUSTOMERS_FILE, POLICIES_FILE, GUIDELINES_FILE):
        (tmp_path / name).write_text("{not valid json", encoding="utf-8")
    return tmp_path


class TestReasonIsEnforced:
    """A blank reason defeats the trace-rationale guarantee -> invalid_input."""

    @pytest.mark.parametrize(
        "call",
        [
            lambda: get_customer("CUST-001", "  "),
            lambda: search_policies("billing_complaint", ""),
            lambda: get_cs_guidelines("bug_report", "   "),
        ],
    )
    def test_blank_reason_rejected(self, call):
        r = call()
        assert r.status is ToolStatus.INVALID_INPUT
        assert "reason" in r.message


class TestEnvelopeContract:
    def test_success_requires_data(self):
        with pytest.raises(ValidationError):
            ToolResponse(
                status=ToolStatus.SUCCESS, data=None, source_ids=[], message=None
            )

    def test_failure_requires_message(self):
        with pytest.raises(ValidationError):
            ToolResponse(
                status=ToolStatus.NOT_FOUND, data=None, source_ids=[], message=None
            )

    def test_failure_must_not_carry_data_or_source_ids(self):
        with pytest.raises(ValidationError):
            ToolResponse(
                status=ToolStatus.NOT_FOUND,
                data={"customer_id": "CUST-001"},
                source_ids=["CUST-001"],
                message="not found",
            )


class TestGetCustomer:
    def test_success(self):
        r = get_customer("CUST-001", REASON)
        assert r.status is ToolStatus.SUCCESS
        assert r.data["name"] == "Alice Nguyen"
        assert r.source_ids == ["CUST-001"]

    def test_not_found(self):
        r = get_customer("CUST-404", REASON)
        assert r.status is ToolStatus.NOT_FOUND
        assert "CUST-404" in r.message

    def test_invalid_input_empty_id(self):
        r = get_customer("   ", REASON)
        assert r.status is ToolStatus.INVALID_INPUT

    def test_tool_error_on_corrupt_data(self, broken_data_dir):
        r = get_customer("CUST-001", REASON, data_dir=broken_data_dir)
        assert r.status is ToolStatus.TOOL_ERROR

    def test_tool_error_on_missing_file(self, tmp_path):
        r = get_customer("CUST-001", REASON, data_dir=tmp_path)
        assert r.status is ToolStatus.TOOL_ERROR

    def test_tool_error_message_does_not_leak_paths(self, tmp_path):
        # The message goes back to the LLM: file name yes, filesystem path no.
        r = get_customer("CUST-001", REASON, data_dir=tmp_path)
        assert str(tmp_path) not in r.message
        assert CUSTOMERS_FILE in r.message

    def test_id_is_whitespace_normalized(self):
        r = get_customer("  CUST-001  ", REASON)
        assert r.status is ToolStatus.SUCCESS


class TestSearchPolicies:
    def test_success_by_category(self):
        r = search_policies("billing_complaint", REASON)
        assert r.status is ToolStatus.SUCCESS
        assert set(r.source_ids) == {"POL-REFUND-001", "POL-SLA-001"}

    def test_query_narrows_results(self):
        r = search_policies("billing_complaint", REASON, query="refund")
        assert r.status is ToolStatus.SUCCESS
        assert r.source_ids == ["POL-REFUND-001"]

    def test_not_found_when_query_matches_nothing(self):
        r = search_policies("billing_complaint", REASON, query="quantum warp drive")
        assert r.status is ToolStatus.NOT_FOUND

    def test_invalid_input_unknown_category(self):
        r = search_policies("spam_report", REASON)
        assert r.status is ToolStatus.INVALID_INPUT
        # actionable message: tells the agent what the valid values are
        assert FeedbackCategory.BILLING_COMPLAINT.value in r.message

    def test_tool_error_on_corrupt_data(self, broken_data_dir):
        r = search_policies("billing_complaint", REASON, data_dir=broken_data_dir)
        assert r.status is ToolStatus.TOOL_ERROR


class TestGetCsGuidelines:
    def test_success(self):
        r = get_cs_guidelines("bug_report", REASON)
        assert r.status is ToolStatus.SUCCESS
        assert r.source_ids == ["SOP-BUG-001"]

    def test_not_found_for_category_without_sop(self):
        r = get_cs_guidelines("other", REASON)
        assert r.status is ToolStatus.NOT_FOUND

    def test_invalid_input_unknown_category(self):
        r = get_cs_guidelines("nonsense", REASON)
        assert r.status is ToolStatus.INVALID_INPUT

    def test_tool_error_on_corrupt_data(self, broken_data_dir):
        r = get_cs_guidelines("bug_report", REASON, data_dir=broken_data_dir)
        assert r.status is ToolStatus.TOOL_ERROR


class TestToolSchemas:
    """The LLM-facing schemas are the contract the agent loop relies on."""

    def test_definitions_and_handlers_align(self):
        assert {d["name"] for d in TOOL_DEFINITIONS} == set(TOOL_HANDLERS)

    def test_reason_is_required_everywhere(self):
        for d in TOOL_DEFINITIONS:
            assert "reason" in d["input_schema"]["required"], d["name"]
            assert "reason" in d["input_schema"]["properties"], d["name"]

    def test_category_schemas_pin_the_taxonomy(self):
        expected = sorted(c.value for c in FeedbackCategory)
        for d in TOOL_DEFINITIONS:
            props = d["input_schema"]["properties"]
            if "category" in props:
                assert sorted(props["category"]["enum"]) == expected, d["name"]

    def test_every_response_is_json_serializable(self):
        # Tool results are fed back to the LLM as JSON text.
        for response in (
            get_customer("CUST-001", REASON),
            search_policies("billing_complaint", REASON),
            get_cs_guidelines("bug_report", REASON),
        ):
            json.dumps(response.model_dump(mode="json"))
