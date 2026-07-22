"""Pipeline-wide constants.

MODEL_ID targets claude-opus-4-8. Note: this model family removed sampling
parameters (temperature/top_p/top_k return HTTP 400), so determinism is
approached via structured outputs + deterministic post-processing rules in
code, not via temperature.
"""

MODEL_ID = "claude-opus-4-8"

CLASSIFY_MAX_TOKENS = 1024
AGENT_MAX_TOKENS = 2048
REPORT_MAX_TOKENS = 4096

# Timeout for LLM API calls only; local JSON-backed tools need none.
LLM_TIMEOUT_SECONDS = 60.0

# Hard cap on LLM calls in the retrieval loop; hitting it exits with
# partial context rather than looping forever.
MAX_TOOL_ITERATIONS = 6

# Transient LLM API errors (timeout, rate limit, 5xx) are retried this many
# times in total with exponential backoff. The pipeline owns this policy;
# the SDK's built-in retries are disabled so attempts aren't multiplied.
LLM_MAX_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 1.0

# classification.confidence below this forces is_ambiguous=True (code-owned
# rule; the LLM's own is_ambiguous flag is only ever OR-ed in, never trusted
# to clear the flag).
AMBIGUITY_THRESHOLD = 0.65

# report.confidence below this forces human review (distinct from the
# classification threshold: a confident classification with poor retrieval
# still yields a low-confidence report).
REPORT_CONFIDENCE_THRESHOLD = 0.70
