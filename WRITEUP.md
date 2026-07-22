# Design Write-up

ch1nhcp (chinhdev.work@gmail.com) — July 2026

## Why this structure

**Single agent inside a controlled pipeline.** The stages of this problem
(classify → gather context → validate → report) are fixed and known, so the
pipeline is fixed code. Agency is applied only where it pays: inside the
retrieval loop, the LLM freely decides which tools to call, in what order,
with what arguments — that is the part that genuinely benefits from
model judgment (e.g. narrowing a policy search, skipping an impossible
lookup). Everything around it is deterministic: a per-source state machine
decides when the loop is done, a two-phase gate decides what needs human
review, and code — never the LLM — sets `status` and `needs_human_review`.
Concretely, the LLM-facing schemas (`Classification`, `ReportDraft`)
contain no control fields at all, so the model cannot set what its schema
cannot express. Multi-agent orchestration would add coordination cost
without adding capability here.

**Classification as a dedicated call, not folded into the loop.** Tradeoff:
one extra LLM call and the risk of an early wrong label steering retrieval.
In exchange: end-to-end schema enforcement (structured output + Pydantic
re-validation + one repair retry), a deterministic ambiguity rule applied
*before* any tool runs (confidence < 0.65 forces review), and a category
that then directs retrieval. The wrong-label risk is contained because low
confidence forces human review anyway.

**Direct Anthropic SDK, no framework.** The value of this exercise is the
control layer — state machine, dedupe, grounding gate, retry taxonomy. A
framework would hide exactly those decisions behind its own abstractions;
the manual tool-calling loop is ~100 lines and every guardrail in it is
explicit and testable. All 181 tests run offline against scripted fakes.

**Grounding as an assertion, not an aspiration.** The gate checks that
every reference ID, every action's `source_ids`, and every ID mentioned in
prose is traceable to retrieved data; violations are stripped, warned, and
force review. "No citation" is only legal for `manual_triage`/`log_only`
actions — checked by enum, not by parsing text.

## What production-ready would add

- **Persistence & queueing**: an ingestion queue, a report store, and a
  review inbox ordered by urgency; review decisions persisted with
  reviewer identity and RBAC on who may approve which action types; only
  approved/overridden actions dispatched to ticketing/refund systems.
- **Reliability**: idempotency keys per submission, dead-letter queue for
  `processing_failed` reports, per-stage (not cumulative) retry metrics.
- **Privacy**: PII redaction before persistence, no long-term storage of
  raw traces (in this demo the trace deliberately keeps everything —
  inspectable reasoning is an assessment requirement).
- **Evaluation**: a labeled test set for classification accuracy and report
  grounding; reviewer overrides fed back as evaluation data; regression
  runs on prompt or model changes.
- **Cost/latency**: prompt caching for the static tool/system prefixes,
  routing classification to a smaller model, batching for non-urgent
  channels.

## Deliberately left out

- **Vector store**: three small JSON files with exact-match/keyword lookup
  are sufficient and keep grounding checks trivially verifiable.
- **Multi-agent**: no independent workstreams to parallelize.
- **UI / deployment / auth**: explicitly out of scope in the brief.
- **Labeled eval set**: listed as bonus; skipped for time.

## Known limitations (accepted knowingly)

- The trace counts transient API retries but not schema repair retries,
  and does not record the agent's free-text turns (each tool call's
  mandatory `reason` carries the rationale instead).
- `retry_count` is cumulative across stages rather than per-stage.
- The `other` category has no SOP by design: uncategorizable feedback
  always lands in human review via the missing-context path.
