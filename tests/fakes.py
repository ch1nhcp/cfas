"""Shared test doubles for the Anthropic client.

Responses are plain SimpleNamespace objects shaped like the SDK's Message
(content blocks with .type, stop_reason) - enough for the code under test.
"""

from types import SimpleNamespace

import anthropic


class FakeRateLimit(anthropic.RateLimitError):
    """Constructor deliberately bypassed - only the type matters."""

    def __init__(self):
        Exception.__init__(self, "rate limited")


class FakeServerError(anthropic.InternalServerError):
    def __init__(self):
        Exception.__init__(self, "overloaded")


class FakeBadRequest(anthropic.BadRequestError):
    def __init__(self):
        Exception.__init__(self, "bad request")


class FakeAuthError(anthropic.AuthenticationError):
    def __init__(self):
        Exception.__init__(self, "invalid api key")


class FakeClient:
    """Returns queued responses in order and records every request.
    A queued Exception instance is raised instead of returned."""

    def __init__(self, responses):
        self.requests = []
        self._responses = list(responses)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name, arguments, call_id="toolu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=arguments, id=call_id)


def make_response(*blocks, stop_reason="end_turn"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason)


def make_text_response(text, stop_reason="end_turn"):
    return make_response(text_block(text), stop_reason=stop_reason)
