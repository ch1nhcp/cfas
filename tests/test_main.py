"""CLI layer tests: argument parsing and exit-code / output behavior."""

import json

import pytest

from cfas.main import main, parse_args


class TestParseArgs:
    def test_parses_full_invocation(self):
        args = parse_args(
            ["App crashes on login", "--customer-id", "CUST-001", "--channel", "email"]
        )
        assert args.feedback_text == "App crashes on login"
        assert args.customer_id == "CUST-001"
        assert args.channel == "email"

    def test_customer_id_defaults_to_none(self):
        args = parse_args(["Great product!", "--channel", "web_form"])
        assert args.customer_id is None

    def test_channel_is_required(self):
        with pytest.raises(SystemExit):
            parse_args(["Some feedback"])

    def test_rejects_unknown_channel(self):
        with pytest.raises(SystemExit):
            parse_args(["Some feedback", "--channel", "carrier_pigeon"])


class TestMain:
    def test_valid_submission_prints_json_and_exits_zero(self, capsys):
        exit_code = main(
            ["I was double-charged", "--customer-id", "CUST-001", "--channel", "email"]
        )
        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["feedback_text"] == "I was double-charged"
        assert payload["customer_id"] == "CUST-001"
        assert payload["channel"] == "email"

    def test_whitespace_only_feedback_exits_nonzero_with_message(self, capsys):
        exit_code = main(["   ", "--channel", "email"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "Invalid submission" in captured.err
        assert "feedback_text" in captured.err
