"""CLI layer tests: argument parsing, exit codes, and an end-to-end run
via failure injection (no network needed)."""

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
        assert args.inject_failure is None

    def test_customer_id_defaults_to_none(self):
        args = parse_args(["Great product!", "--channel", "web_form"])
        assert args.customer_id is None

    def test_channel_is_required(self):
        with pytest.raises(SystemExit):
            parse_args(["Some feedback"])

    def test_rejects_unknown_channel(self):
        with pytest.raises(SystemExit):
            parse_args(["Some feedback", "--channel", "carrier_pigeon"])

    def test_rejects_unknown_injection_mode(self):
        with pytest.raises(SystemExit):
            parse_args(["Hi", "--channel", "email", "--inject-failure", "everything"])


class TestMain:
    def test_whitespace_only_feedback_exits_nonzero_with_message(self, capsys):
        exit_code = main(["   ", "--channel", "email"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "Invalid submission" in captured.err
        assert "feedback_text" in captured.err

    def test_injected_llm_failure_runs_end_to_end(self, tmp_path, capsys):
        # exercises the full pipeline offline: retries, failure report,
        # trace on stderr, artifacts on disk, report JSON on stdout
        exit_code = main(
            [
                "I was double-charged",
                "--customer-id",
                "CUST-001",
                "--channel",
                "email",
                "--inject-failure",
                "llm",
                "--out-dir",
                str(tmp_path),
            ]
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        report = json.loads(captured.out)
        assert report["status"] == "processing_failed"
        assert report["needs_human_review"] is True
        assert "injected LLM failure" in report["review_reason"]
        assert "processing_failed" in captured.err  # trace printed
        run_dirs = list(tmp_path.iterdir())
        assert len(run_dirs) == 1
        for name in ("input.json", "trace.json", "report.json"):
            assert (run_dirs[0] / name).exists()
