"""Retry policy tests: transient errors retried with backoff, everything
else raised immediately."""

import anthropic
import pytest
from fakes import (
    FakeBadRequest,
    FakeClient,
    FakeRateLimit,
    FakeServerError,
    make_text_response,
)

from cfas.config import LLM_MAX_ATTEMPTS, RETRY_BASE_DELAY_SECONDS
from cfas.retry import RetryingClient


class TestTransientErrors:
    def test_retried_with_exponential_backoff_then_succeeds(self):
        inner = FakeClient(
            [FakeRateLimit(), FakeServerError(), make_text_response("ok")]
        )
        sleeps = []
        client = RetryingClient(inner, sleep=sleeps.append)
        response = client.messages.create(model="m")
        assert response.content[0].text == "ok"
        assert client.retry_count == 2
        assert sleeps == [
            RETRY_BASE_DELAY_SECONDS,
            RETRY_BASE_DELAY_SECONDS * 2,
        ]
        assert len(inner.requests) == 3

    def test_exhausted_attempts_reraise_last_error(self):
        inner = FakeClient([FakeRateLimit() for _ in range(LLM_MAX_ATTEMPTS)])
        client = RetryingClient(inner, sleep=lambda _s: None)
        with pytest.raises(anthropic.RateLimitError):
            client.messages.create(model="m")
        assert len(inner.requests) == LLM_MAX_ATTEMPTS


class TestNonTransientErrors:
    def test_bad_request_not_retried(self):
        inner = FakeClient([FakeBadRequest()])
        client = RetryingClient(inner, sleep=lambda _s: None)
        with pytest.raises(anthropic.BadRequestError):
            client.messages.create(model="m")
        assert len(inner.requests) == 1
        assert client.retry_count == 0
