"""Retry policy for transient LLM API errors.

Retried (with exponential backoff): timeouts, connection errors, rate
limits, 5xx/overloaded. NOT retried: invalid API key, invalid request,
schema failures (ClassificationError/ReportGenerationError carry those),
content-policy refusals. Local tool errors are deterministic and never
routed through here at all.

The SDK's own retries are disabled at client construction (see cfas.llm)
so this policy is the single source of truth for attempt counts.
"""

import time
from collections.abc import Callable
from types import SimpleNamespace

import anthropic

from cfas.config import LLM_MAX_ATTEMPTS, RETRY_BASE_DELAY_SECONDS

# APITimeoutError subclasses APIConnectionError; InternalServerError covers
# every >=500 status including 529 overloaded.
TRANSIENT_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


class RetryingClient:
    """Anthropic-shaped adapter adding transparent retries to
    messages.create. retry_count is exposed for the trace."""

    def __init__(
        self,
        inner,
        max_attempts: int = LLM_MAX_ATTEMPTS,
        base_delay: float = RETRY_BASE_DELAY_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._inner = inner
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._sleep = sleep
        self.retry_count = 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._inner.messages.create(**kwargs)
            except TRANSIENT_ERRORS:
                if attempt == self._max_attempts:
                    raise
                self.retry_count += 1
                self._sleep(self._base_delay * 2 ** (attempt - 1))
        raise AssertionError("unreachable")  # loop always returns or raises
