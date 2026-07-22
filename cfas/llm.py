"""Anthropic client factory.

The timeout applies to LLM API calls only - local retrieval tools read JSON
in-process and need none. Credentials resolve from the environment
(ANTHROPIC_API_KEY); the SDK raises a clear error when absent.
"""

import anthropic

from cfas.config import LLM_TIMEOUT_SECONDS


def default_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(timeout=LLM_TIMEOUT_SECONDS)
