"""Trace recorder and artifact-writing tests."""

import io
import json

from cfas.trace import TraceRecorder, print_trace_summary, write_run_artifacts


class TestTraceRecorder:
    def test_records_ordered_steps_with_elapsed_time(self):
        recorder = TraceRecorder("RUN-test")
        recorder.record("intake", channel="email")
        recorder.record("final", status="pending_review")
        trace = recorder.to_dict()
        assert trace["run_id"] == "RUN-test"
        assert [s["step"] for s in trace["steps"]] == ["intake", "final"]
        elapsed = [s["elapsed_ms"] for s in trace["steps"]]
        assert all(ms >= 0 for ms in elapsed)
        assert elapsed == sorted(elapsed)

    def test_to_dict_is_json_serializable(self):
        recorder = TraceRecorder("RUN-test")
        recorder.record("step", data={"nested": [1, 2]})
        json.dumps(recorder.to_dict())


class TestPrintTraceSummary:
    def test_prints_run_id_and_steps(self):
        recorder = TraceRecorder("RUN-abc")
        recorder.record("classification", category="bug_report")
        out = io.StringIO()
        print_trace_summary(recorder.to_dict(), stream=out)
        rendered = out.getvalue()
        assert "RUN-abc" in rendered
        assert "classification" in rendered
        assert "bug_report" in rendered

    def test_long_details_are_truncated(self):
        recorder = TraceRecorder("RUN-abc")
        recorder.record("big", blob="x" * 1000)
        out = io.StringIO()
        print_trace_summary(recorder.to_dict(), stream=out)
        assert "..." in out.getvalue()


class TestWriteRunArtifacts:
    def test_writes_three_json_files(self, tmp_path):
        run_dir = write_run_artifacts(
            run_id="RUN-xyz",
            submission_json={"feedback_text": "hi"},
            trace={"run_id": "RUN-xyz", "steps": []},
            report_json={"report_id": "RPT-1"},
            out_dir=tmp_path,
        )
        assert run_dir == tmp_path / "RUN-xyz"
        for name in ("input.json", "trace.json", "report.json"):
            with open(run_dir / name, encoding="utf-8") as f:
                assert json.load(f)
