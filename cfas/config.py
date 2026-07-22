"""Pipeline-wide constants.

MODEL_ID targets claude-opus-4-8. Note: this model family removed sampling
parameters (temperature/top_p/top_k return HTTP 400), so determinism is
approached via structured outputs + deterministic post-processing rules in
code, not via temperature.
"""

MODEL_ID = "claude-opus-4-8"

CLASSIFY_MAX_TOKENS = 1024

# classification.confidence below this forces is_ambiguous=True (code-owned
# rule; the LLM's own is_ambiguous flag is only ever OR-ed in, never trusted
# to clear the flag).
AMBIGUITY_THRESHOLD = 0.65
