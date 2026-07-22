"""Shared test doubles for the Anthropic client.

Responses are plain SimpleNamespace objects shaped like the SDK's Message
(content blocks with .type, stop_reason) - enough for the code under test.
"""

from types import SimpleNamespace


class FakeClient:
    """Returns queued responses in order and records every request."""

    def __init__(self, responses):
        self.requests = []
        self._responses = list(responses)
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self._responses.pop(0)


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name, arguments, call_id="toolu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=arguments, id=call_id)


def make_response(*blocks, stop_reason="end_turn"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason)


def make_text_response(text, stop_reason="end_turn"):
    return make_response(text_block(text), stop_reason=stop_reason)
