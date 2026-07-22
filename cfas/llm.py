"""LLM plumbing shared by every pipeline stage.

- default_client(): Anthropic client. The timeout applies to LLM API calls
  only - local retrieval tools read JSON in-process and need none.
  Credentials resolve from the environment (ANTHROPIC_API_KEY).
- structured_output_schema(): Pydantic schema -> wire schema for
  output_config.format. Numeric/length constraints are unsupported there,
  so they are stripped and enforced client-side by Pydantic instead.
- structured_llm_call(): the create -> validate -> one-repair-retry pattern
  shared by the classification and report stages.
"""

from typing import TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from cfas.config import LLM_TIMEOUT_SECONDS
from cfas.models import FeedbackSubmission

T = TypeVar("T", bound=BaseModel)


def render_submission_block(
    submission: FeedbackSubmission, customer_line: str | None = None
) -> str:
    """Uniform prompt rendering of a submission: metadata plus the feedback
    text wrapped as untrusted content (every stage prompt shares this)."""
    customer = customer_line or submission.customer_id or "not provided"
    return (
        f"Channel: {submission.channel.value}\n"
        f"Customer ID: {customer}\n"
        f"Received at: {submission.timestamp.isoformat()}\n\n"
        f"<customer_feedback>\n{submission.feedback_text}\n</customer_feedback>"
    )


def default_client() -> anthropic.Anthropic:
    # max_retries=0: the pipeline's RetryingClient owns the retry policy;
    # leaving the SDK default (2) would multiply attempts.
    return anthropic.Anthropic(timeout=LLM_TIMEOUT_SECONDS, max_retries=0)


_UNSUPPORTED_SCHEMA_KEYS = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
}


def _strip_unsupported(node: object) -> object:
    """Drop constraint keys the structured-output API rejects and forbid
    extra properties on every object. (Key-name based: safe because no
    schema property shares a name with a constraint keyword.)"""
    if isinstance(node, dict):
        cleaned = {
            key: _strip_unsupported(value)
            for key, value in node.items()
            if key not in _UNSUPPORTED_SCHEMA_KEYS
        }
        if cleaned.get("type") == "object":
            cleaned["additionalProperties"] = False
        return cleaned
    if isinstance(node, list):
        return [_strip_unsupported(item) for item in node]
    return node


def structured_output_schema(model_cls: type[BaseModel]) -> dict:
    return _strip_unsupported(model_cls.model_json_schema())


def _extract_json_text(response, error_cls: type[Exception]) -> str:
    if response.stop_reason == "refusal":
        raise error_cls("model refused to process this submission")
    if response.stop_reason == "max_tokens":
        raise error_cls("structured output truncated (max_tokens)")
    for block in response.content:
        if block.type == "text":
            return block.text
    raise error_cls("model response contained no text block")


def _repair_request(request: dict, raw_output: str, error: ValidationError) -> dict:
    problems = "; ".join(
        f"{'.'.join(str(part) for part in e['loc'])}: {e['msg']}"
        for e in error.errors()
    )
    return {
        **request,
        "messages": [
            *request["messages"],
            {"role": "assistant", "content": raw_output},
            {
                "role": "user",
                "content": (
                    "Your previous output failed validation: "
                    f"{problems}. Return a corrected JSON object that "
                    "satisfies the schema exactly."
                ),
            },
        ],
    }


def structured_llm_call(
    client,
    request: dict,
    model_cls: type[T],
    error_cls: type[Exception],
) -> T:
    """Run one structured-output call, validating with Pydantic.

    On validation failure, exactly ONE repair retry echoes the invalid
    output plus the validation errors back to the model; a second failure
    raises error_cls. Refusal/truncation raise immediately (they are not
    schema failures). Transient API errors propagate to the caller.
    """
    response = client.messages.create(**request)
    raw_output = _extract_json_text(response, error_cls)
    try:
        return model_cls.model_validate_json(raw_output)
    except ValidationError as first_error:
        repair_response = client.messages.create(
            **_repair_request(request, raw_output, first_error)
        )
        repaired_output = _extract_json_text(repair_response, error_cls)
        try:
            return model_cls.model_validate_json(repaired_output)
        except ValidationError as second_error:
            raise error_cls(
                "structured output failed validation after one repair retry: "
                f"{second_error.error_count()} error(s)"
            ) from second_error
