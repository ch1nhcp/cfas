# CFAS — Agentic Customer Feedback System

An agentic pipeline that turns free-text customer feedback into an
actionable, grounded triage report for a CS officer: classify → gather
context via tool-calling → validate → report → human review.

Design principle throughout: **the LLM proposes, code controls.** The agent
decides *how* to gather context (which tools, what order, what arguments);
deterministic code owns every control decision (review flags, report status,
grounding, retries, loop bounds).

- Design rationale and tradeoffs: [WRITEUP.md](WRITEUP.md)
- Author: ch1nhcp (chinhdev.work@gmail.com) — July 2026

## Architecture

```
 feedback text + metadata (CLI)
          │
          ▼
 ┌─ intake ──────────┐  validate & normalize          intake.py, models.py
 └────────┬──────────┘
          ▼
 ┌─ classification ──┐  LLM call #1, structured       classify.py
 │                   │  output → Classification;
 │                   │  1 repair retry; code rule:
 │                   │  confidence < 0.65 → ambiguous
 └────────┬──────────┘
          ▼
 ┌─ retrieval loop ──┐  ≤ 6 LLM turns; the agent      agent.py, tools.py
 │  get_customer     │  picks tools/order/args;
 │  search_policies  │  per-source state machine,
 │  get_cs_guidelines│  dedupe cache, one nudge,
 └────────┬──────────┘  one final-chance turn
          ▼
 ┌─ gate phase 1 ────┐  validate_context():           gate.py
 └────────┬──────────┘  deterministic review rules
          ▼
 ┌─ report draft ────┐  final LLM call → ReportDraft  report.py
 └────────┬──────────┘  (schema has NO control fields)
          ▼
 ┌─ gate phase 2 ────┐  validate_report(): grounding
 └────────┬──────────┘  assertion, strip + flag
          ▼
   FeedbackReport (status=pending_review, code-assembled)
          │
          ▼
 ┌─ human review ────┐  approve / override / reject   review.py (stub)
 └───────────────────┘
```

Cross-cutting: `retry.py` (backoff for transient LLM errors wraps every LLM
call), `trace.py` (every run records all intermediate steps),
`pipeline.py` (orchestration + `make_failure_report` fallback).

## Setup

Requires Python ≥ 3.11 and an Anthropic API key.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run one feedback

```bash
.venv/bin/python -m cfas.main \
  "I was charged twice for my subscription this month" \
  --customer-id CUST-001 --channel email
```

- **stdout**: the final report JSON (machine-readable)
- **stderr**: the step-by-step trace (reasoning steps, tool calls with the
  agent's stated reason, validation warnings)
- **`runs/<run_id>/`**: `input.json`, `trace.json`, `report.json`

Exit codes: `0` report produced · `1` invalid input · `2` configuration
error (bad API key etc.) · `3` pipeline fell back to a `processing_failed`
report.

## Sample cases

Five sample cases cover the required happy paths and edge cases. Their
recorded runs (one directory per run: `input.json`, `trace.json`,
`report.json`) are kept under `samples/`; each can be (re)generated with
the command in its row (`python` = `.venv/bin/python`):

| # | Case | Demonstrates | Command |
|---|------|--------------|---------|
| 1 | Clear refund complaint, premium customer | Happy path, grounded citations | `python -m cfas.main "I was charged twice for my subscription this month, please refund the duplicate" --customer-id CUST-001 --channel email --out-dir samples` |
| 2 | Severe bug, enterprise customer | High urgency → automatic review | `python -m cfas.main "Webhook deliveries have been down for 6 hours, our whole logistics operation is blocked!" --customer-id CUST-002 --channel chat --out-dir samples` |
| 3 | Unknown customer ID | Missing record → `missing_context`, review | `python -m cfas.main "Where is my refund?" --customer-id CUST-404 --channel email --out-dir samples` |
| 4 | Vague / multi-topic feedback | Ambiguous classification → forced review | `python -m cfas.main "billing is weird and the app sometimes does the thing wrong?" --customer-id CUST-004 --channel web_form --out-dir samples` |
| 5 | Simulated LLM outage | Retry + `processing_failed` fallback report | `python -m cfas.main "I was double-charged" --customer-id CUST-001 --channel email --inject-failure llm --out-dir samples` |

`--inject-failure tool` additionally simulates a broken local data source
(every tool returns `tool_error`; the report is produced with everything
flagged as missing and routed to review).

## Classification taxonomy

`bug_report`, `billing_complaint`, `feature_request`, `praise`,
`churn_risk`, `abuse_policy_violation`, `other` — plus sentiment
(positive/neutral/negative), urgency (low/medium/high), confidence (0–1),
and an ambiguity flag. Code forces `is_ambiguous=true` whenever confidence
< 0.65; the LLM can raise the flag but never clear it.

## Tools

Three retrieval tools over mock JSON data (`data/`), each requiring a
`reason` argument so the trace always carries decision rationale:

| Tool | Looks up | Source IDs |
|---|---|---|
| `get_customer(customer_id, reason)` | tier, tenure, tickets, orders | `CUST-…` |
| `search_policies(category, reason, query?)` | applicable company policies | `POL-…` |
| `get_cs_guidelines(category, reason)` | the CS workflow SOP | `SOP-…` |

All tools return one envelope `{status, data, source_ids, message}` with
`status ∈ {success, not_found, invalid_input, tool_error}` — each status
maps to a distinct agent behavior (`not_found`: record the gap, never
invent data; `invalid_input`: fix arguments and retry; `tool_error`:
deterministic local failure, no retry).

## Error handling & guardrails

- **Retrieval loop**: hard cap of 6 LLM turns; duplicate calls served from
  a dedupe cache (the `reason` field is excluded from the key); per-source
  state machine `pending → retrieved | not_found | unavailable |
  tool_error`; anonymous submissions mark the customer source
  `unavailable` up front; a stalled agent gets exactly one nudge.
- **Grounding gate** (deterministic, no LLM): every reference ID and every
  action's `source_ids` must be traceable to retrieved data — violations
  are stripped, warned about, and force human review; hallucinated IDs in
  prose are caught too (case-insensitive, multi-segment variants included).
  Empty citations are legal only for `manual_triage`/`log_only` actions.
- **Retries**: exponential backoff for transient LLM API errors only
  (timeout, rate limit, 5xx). Schema failures get exactly one repair retry
  with the validation errors echoed back. Configuration errors (bad key,
  malformed request) crash loudly instead of producing failure reports.
- **Terminal fallback**: if the pipeline cannot produce a report, a valid
  `processing_failed` report is emitted and routed to manual triage — the
  pipeline never returns nothing.

## Human review

`review.py` is a deliberate stub (reachable from tests, not wired into the
CLI): `apply_review(report, decision, reviewer_id, note,
overridden_actions)` implements `pending_review → approved | overridden |
rejected` and returns the updated report plus an audit record. The
production flow it stands in for is described in its module docstring and
in [WRITEUP.md](WRITEUP.md).

```python
from cfas.review import ReviewDecision, apply_review
updated, record = apply_review(report, ReviewDecision.APPROVE, reviewer_id="cs-7")
```

## Tests

```bash
.venv/bin/python -m pytest              # 181 tests, no network needed
.venv/bin/python -m pytest --cov=cfas   # coverage (99%)
```

The whole suite runs offline: every LLM interaction is scripted through
fakes; the tools run against the real `data/` files.

## Project layout

```
cfas/
  models.py    frozen Pydantic contracts (LLM-facing vs code-owned split)
  intake.py    input boundary
  classify.py  classification stage        llm.py     shared LLM plumbing
  tools.py     retrieval tools + schemas   retry.py   transient-error policy
  agent.py     bounded tool-calling loop   trace.py   observable trace
  gate.py      two-phase validation gate   pipeline.py orchestration
  report.py    report generation/assembly  review.py  human-review stub
  main.py      CLI
data/          mock datasets (citable IDs)
tests/         pytest suite (offline)
samples/       5 recorded sample runs
```

## AI assistance note

Built with Claude Code (Claude Fable 5) used as a pair programmer under
close direction: the build plan was written and reviewed first; each step
was implemented test-first, then passed through automated two-axis code
review (standards + spec fidelity) before commit, plus a final whole-repo
review (three parallel reviewers) whose findings were all addressed. All
code, tests, and documentation in this repository went through that loop.
