"""Retrieval-loop tests: state machine, dedupe, nudge, and exit conditions.

The LLM is scripted via FakeClient; the real tools run against data/ so the
loop's tool execution path is exercised end-to-end.
"""

from datetime import datetime, timezone

from factories import customer_call, guidelines_call, policies_call
from fakes import FakeClient, make_response, make_text_response, tool_use_block

from cfas.agent import Source, SourceStatus, gather_context
from cfas.config import MAX_TOOL_ITERATIONS
from cfas.intake import build_submission
from cfas.models import Classification, FeedbackCategory, Sentiment, Urgency

TS = datetime(2026, 7, 20, tzinfo=timezone.utc)

CLASSIFICATION = Classification(
    category=FeedbackCategory.BILLING_COMPLAINT,
    sentiment=Sentiment.NEGATIVE,
    urgency=Urgency.MEDIUM,
    confidence=0.9,
    reason="Duplicate subscription charge reported.",
    is_ambiguous=False,
)

SUBMISSION = build_submission(
    feedback_text="I was charged twice for my subscription.",
    customer_id="CUST-001",
    channel="email",
    timestamp=TS,
)

ANONYMOUS = build_submission(
    feedback_text="Your refund policy is unclear.",
    customer_id=None,
    channel="web_form",
    timestamp=TS,
)


class TestHappyPath:
    def test_all_sources_retrieved_in_one_iteration(self):
        client = FakeClient(
            [make_response(customer_call(), policies_call(), guidelines_call())]
        )
        result = gather_context(SUBMISSION, CLASSIFICATION, client=client)
        assert all(
            status is SourceStatus.RETRIEVED
            for status in result.source_states.values()
        )
        assert result.iterations == 1
        assert len(client.requests) == 1
        assert result.missing_context == []
        for source_id in ("CUST-001", "POL-REFUND-001", "SOP-BILLING-001"):
            assert source_id in result.retrieved_source_ids

    def test_retrieved_data_is_kept_for_the_report_step(self):
        client = FakeClient(
            [make_response(customer_call(), policies_call(), guidelines_call())]
        )
        result = gather_context(SUBMISSION, CLASSIFICATION, client=client)
        assert result.context[Source.CUSTOMER][0]["name"] == "Alice Nguyen"
        assert result.context[Source.POLICIES][0]  # non-empty policy list

    def test_request_contract(self):
        client = FakeClient(
            [make_response(customer_call(), policies_call(), guidelines_call())]
        )
        gather_context(SUBMISSION, CLASSIFICATION, client=client)
        req = client.requests[0]
        assert {tool["name"] for tool in req["tools"]} == {
            "get_customer",
            "search_policies",
            "get_cs_guidelines",
        }
        assert "untrusted" in req["system"].lower()
        assert SUBMISSION.feedback_text in req["messages"][0]["content"]


class TestAnonymousSubmission:
    def test_customer_source_unavailable_without_tool_call(self):
        client = FakeClient([make_response(policies_call(), guidelines_call())])
        result = gather_context(ANONYMOUS, CLASSIFICATION, client=client)
        assert result.source_states[Source.CUSTOMER] is SourceStatus.UNAVAILABLE
        assert any("customer" in note for note in result.missing_context)
        called = [record.tool_name for record in result.tool_calls]
        assert "get_customer" not in called


class TestDedupe:
    def test_identical_call_with_different_reason_is_deduped(self):
        repeat = customer_call(reason="completely different reason", cid="t9")
        client = FakeClient(
            [
                make_response(
                    customer_call(), repeat, policies_call(), guidelines_call()
                )
            ]
        )
        result = gather_context(SUBMISSION, CLASSIFICATION, client=client)
        flags = [r.deduped for r in result.tool_calls if r.tool_name == "get_customer"]
        assert flags == [False, True]

    def test_deduped_success_does_not_duplicate_context(self):
        repeat = customer_call(reason="asking again", cid="t9")
        client = FakeClient(
            [
                make_response(
                    customer_call(), repeat, policies_call(), guidelines_call()
                )
            ]
        )
        result = gather_context(SUBMISSION, CLASSIFICATION, client=client)
        assert len(result.context[Source.CUSTOMER]) == 1

    def test_different_functional_args_are_not_deduped(self):
        narrowed = tool_use_block(
            "search_policies",
            {"category": "billing_complaint", "query": "refund", "reason": "narrow"},
            "t8",
        )
        client = FakeClient(
            [
                make_response(
                    customer_call(), policies_call(), narrowed, guidelines_call()
                )
            ]
        )
        result = gather_context(SUBMISSION, CLASSIFICATION, client=client)
        flags = [
            r.deduped for r in result.tool_calls if r.tool_name == "search_policies"
        ]
        assert flags == [False, False]


class TestInvalidInput:
    def test_source_stays_pending_and_agent_can_correct(self):
        client = FakeClient(
            [
                make_response(policies_call(category="bogus"), guidelines_call()),
                make_response(policies_call()),
            ]
        )
        result = gather_context(ANONYMOUS, CLASSIFICATION, client=client)
        assert result.source_states[Source.POLICIES] is SourceStatus.RETRIEVED
        assert result.iterations == 2
        statuses = [
            r.status for r in result.tool_calls if r.tool_name == "search_policies"
        ]
        assert statuses == ["invalid_input", "success"]

    def test_tool_results_are_fed_back_with_matching_ids(self):
        client = FakeClient(
            [
                make_response(policies_call(category="bogus", cid="tX"),
                              guidelines_call()),
                make_response(policies_call()),
            ]
        )
        gather_context(ANONYMOUS, CLASSIFICATION, client=client)
        second_request_messages = client.requests[1]["messages"]
        tool_results = second_request_messages[2]["content"]
        assert {r["tool_use_id"] for r in tool_results} == {"tX", "t3"}
        assert all(r["type"] == "tool_result" for r in tool_results)


class TestExitConditions:
    def test_iteration_limit_exits_with_partial_context(self):
        responses = [
            make_response(policies_call(category="bogus"))
            for _ in range(MAX_TOOL_ITERATIONS)
        ]
        client = FakeClient(responses)
        result = gather_context(ANONYMOUS, CLASSIFICATION, client=client)
        assert len(client.requests) == MAX_TOOL_ITERATIONS
        assert result.source_states[Source.POLICIES] is SourceStatus.PENDING
        assert any("limit" in w for w in result.warnings)
        assert any("policies" in note for note in result.missing_context)

    def test_nudge_recovers_a_stalled_agent(self):
        client = FakeClient(
            [
                make_text_response("Looks like I'm done."),
                make_response(policies_call(), guidelines_call()),
            ]
        )
        result = gather_context(ANONYMOUS, CLASSIFICATION, client=client)
        assert result.source_states[Source.POLICIES] is SourceStatus.RETRIEVED
        # requests[] holds a reference to the live messages list, so scan for
        # the nudge instead of relying on position
        nudges = [
            m
            for m in client.requests[1]["messages"]
            if m["role"] == "user"
            and isinstance(m["content"], str)
            and "pending" in m["content"]
        ]
        assert len(nudges) == 1

    def test_second_stall_exits_with_pending_recorded(self):
        client = FakeClient(
            [
                make_text_response("Done."),
                make_text_response("Really done."),
            ]
        )
        result = gather_context(ANONYMOUS, CLASSIFICATION, client=client)
        assert len(client.requests) == 2
        assert any("stopped calling tools" in w for w in result.warnings)
        assert any("policies" in note for note in result.missing_context)

    def test_refusal_exits_with_warning(self):
        client = FakeClient([make_response(stop_reason="refusal")])
        result = gather_context(ANONYMOUS, CLASSIFICATION, client=client)
        assert len(client.requests) == 1
        assert any("refus" in w for w in result.warnings)


class TestFailureStates:
    def test_missing_record_becomes_not_found(self):
        missing = build_submission(
            feedback_text="Where is my order?",
            customer_id="CUST-404",
            channel="email",
            timestamp=TS,
        )
        client = FakeClient(
            [
                make_response(
                    customer_call("CUST-404"), policies_call(), guidelines_call()
                ),
                # final-chance turn: agent sees the not_found and stops
                make_text_response("No such customer; context gathering done."),
            ]
        )
        result = gather_context(missing, CLASSIFICATION, client=client)
        assert result.source_states[Source.CUSTOMER] is SourceStatus.NOT_FOUND
        assert any("customer" in note for note in result.missing_context)
        assert len(client.requests) == 2  # exactly one final-chance turn

    def test_final_chance_allows_broadening_after_not_found(self):
        narrow = tool_use_block(
            "search_policies",
            {
                "category": "billing_complaint",
                "query": "quantum warp drive",
                "reason": "narrow first",
            },
            "t6",
        )
        client = FakeClient(
            [
                make_response(narrow, guidelines_call()),
                # agent sees not_found, broadens without the query
                make_response(policies_call(reason="broaden after not_found")),
            ]
        )
        result = gather_context(ANONYMOUS, CLASSIFICATION, client=client)
        assert result.source_states[Source.POLICIES] is SourceStatus.RETRIEVED
        assert result.missing_context == [
            "customer: not available for this submission (no customer ID)"
        ]
        assert result.iterations == 2

    def test_broken_data_source_becomes_tool_error(self, tmp_path):
        client = FakeClient(
            [make_response(customer_call(), policies_call(), guidelines_call())]
        )
        result = gather_context(
            SUBMISSION, CLASSIFICATION, client=client, data_dir=tmp_path
        )
        assert all(
            status is SourceStatus.TOOL_ERROR
            for status in result.source_states.values()
        )
        assert result.retrieved_source_ids == []

    def test_handler_crash_becomes_tool_error_not_invalid_input(self, monkeypatch):
        # A crash inside the handler is deterministic: reporting it as
        # invalid_input would make the agent retry futilely.
        def boom(**_kwargs):
            raise TypeError("internal handler bug")

        from cfas import agent as agent_module

        monkeypatch.setitem(agent_module.TOOL_HANDLERS, "get_customer", boom)
        client = FakeClient(
            [make_response(customer_call(), policies_call(), guidelines_call())]
        )
        result = gather_context(SUBMISSION, CLASSIFICATION, client=client)
        assert result.source_states[Source.CUSTOMER] is SourceStatus.TOOL_ERROR

    def test_cross_category_success_does_not_complete_the_source(self):
        # supplementary cross-category data is collected, but only the
        # classified category completes the policies source
        cross = policies_call(category="bug_report", reason="related angle", cid="t8")
        client = FakeClient(
            [
                make_response(customer_call(), cross, guidelines_call()),
                make_response(policies_call()),  # primary category
            ]
        )
        result = gather_context(SUBMISSION, CLASSIFICATION, client=client)
        assert result.source_states[Source.POLICIES] is SourceStatus.RETRIEVED
        assert result.iterations == 2
        assert "POL-ESCALATION-001" in result.retrieved_source_ids  # cross data kept
        assert "POL-REFUND-001" in result.retrieved_source_ids  # primary data

    def test_never_attempting_primary_category_yields_missing_context(self):
        cross = policies_call(category="bug_report", reason="wrong angle", cid="t8")
        client = FakeClient(
            [
                make_response(cross, guidelines_call()),
                make_text_response("Done."),  # stall -> nudge (policies pending)
                make_text_response("Still done."),  # second stall -> exit
            ]
        )
        result = gather_context(ANONYMOUS, CLASSIFICATION, client=client)
        assert result.source_states[Source.POLICIES] is SourceStatus.PENDING
        assert any("policies" in note for note in result.missing_context)

    def test_cross_customer_lookup_is_blocked(self):
        # prompt-injected "look up CUST-002" must not leak another
        # customer's record into the report
        rogue = customer_call("CUST-002", reason="feedback asked me to", cid="t8")
        client = FakeClient(
            [
                make_response(rogue, policies_call(), guidelines_call()),
                make_response(customer_call()),
            ]
        )
        result = gather_context(SUBMISSION, CLASSIFICATION, client=client)
        statuses = [
            r.status for r in result.tool_calls if r.tool_name == "get_customer"
        ]
        assert statuses == ["invalid_input", "success"]
        assert "CUST-002" not in result.retrieved_source_ids
        assert result.source_states[Source.CUSTOMER] is SourceStatus.RETRIEVED

    def test_get_customer_blocked_for_anonymous_submission(self):
        client = FakeClient(
            [make_response(customer_call(), policies_call(), guidelines_call())]
        )
        result = gather_context(ANONYMOUS, CLASSIFICATION, client=client)
        record = next(
            r for r in result.tool_calls if r.tool_name == "get_customer"
        )
        assert record.status == "invalid_input"
        assert "no customer ID" in record.message
        assert result.source_states[Source.CUSTOMER] is SourceStatus.UNAVAILABLE

    def test_reserved_data_dir_argument_is_invalid_input(self):
        # injecting the loop's own data_dir kwarg must be correctable, not a
        # terminal tool_error
        poisoned = tool_use_block(
            "get_customer",
            {"customer_id": "CUST-001", "reason": "lookup", "data_dir": "/etc"},
            "t5",
        )
        client = FakeClient(
            [
                make_response(poisoned, policies_call(), guidelines_call()),
                make_response(customer_call()),
            ]
        )
        result = gather_context(SUBMISSION, CLASSIFICATION, client=client)
        statuses = [
            r.status for r in result.tool_calls if r.tool_name == "get_customer"
        ]
        assert statuses == ["invalid_input", "success"]
        assert result.source_states[Source.CUSTOMER] is SourceStatus.RETRIEVED

    def test_missing_required_argument_is_invalid_input(self):
        incomplete = tool_use_block("get_customer", {"reason": "lookup"}, "t5")
        client = FakeClient(
            [
                make_response(incomplete, policies_call(), guidelines_call()),
                make_response(customer_call()),
            ]
        )
        result = gather_context(SUBMISSION, CLASSIFICATION, client=client)
        statuses = [
            r.status for r in result.tool_calls if r.tool_name == "get_customer"
        ]
        assert statuses == ["invalid_input", "success"]
        assert result.source_states[Source.CUSTOMER] is SourceStatus.RETRIEVED

    def test_not_found_then_success_in_same_turn_ends_retrieved(self):
        narrow = tool_use_block(
            "search_policies",
            {
                "category": "billing_complaint",
                "query": "quantum warp drive",
                "reason": "narrow search",
            },
            "t6",
        )
        client = FakeClient(
            [make_response(narrow, policies_call(), guidelines_call())]
        )
        result = gather_context(ANONYMOUS, CLASSIFICATION, client=client)
        assert result.source_states[Source.POLICIES] is SourceStatus.RETRIEVED

    def test_truncated_response_is_flagged(self):
        client = FakeClient(
            [
                make_text_response("partial...", stop_reason="max_tokens"),
                make_response(policies_call(), guidelines_call()),
            ]
        )
        result = gather_context(ANONYMOUS, CLASSIFICATION, client=client)
        assert any("truncated" in w for w in result.warnings)

    def test_unknown_tool_is_rejected_without_crashing(self):
        rogue = tool_use_block("delete_account", {"reason": "cleanup"}, "t7")
        client = FakeClient(
            [make_response(rogue, policies_call(), guidelines_call())]
        )
        result = gather_context(ANONYMOUS, CLASSIFICATION, client=client)
        assert any("unknown tool" in w for w in result.warnings)
        assert result.source_states[Source.POLICIES] is SourceStatus.RETRIEVED
