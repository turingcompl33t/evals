"""
API helpers.
"""

import json
import time
from typing import Any, Callable, Literal, Sequence, Type, TypeAlias

from openai import OpenAI

Message: TypeAlias = dict[Literal["role", "content"], str]
Messages: TypeAlias = Sequence[Message]


def generate(
    client: OpenAI,
    model: str,
    messages: Messages,
    temperature: float = 1,
    max_tokens: int = 1024,
) -> str:
    """
    Generate a response using the OpenAI API.
    :param model: The name of the model to use (e.g., "gpt-4o-mini").
    :param messages: A list of message dictionaries with 'role' and 'content' keys.
    :param temperature: Controls randomness in output. Higher values make output more random.
    :param max_tokens: The maximum number of tokens to generate.
    :return: The generated response from the model.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content  # type: ignore
    except Exception as e:
        raise RuntimeError(f"error in generation:\n{e}") from e


def generate_structured(
    client: OpenAI,
    model: str,
    messages: Messages,
    response_format: Type,
    temperature: float = 1,
    max_tokens: int = 1024,
    extra_body: dict[str, Any] | None = None,
) -> dict:
    """
    Generate a response using the OpenAI API.
    :param model: The name of the model to use (e.g., "gpt-4o-mini").
    :param messages: A list of message dictionaries with 'role' and 'content' keys.
    :param response_format: The response format
    :param temperature: Controls randomness in output. Higher values make output more random.
    :param max_tokens: The maximum number of tokens to generate.
    :param extra_body: Additional request body fields (e.g. OpenRouter `provider`, `reasoning`).
    :return: The generated response from the model.
    """
    try:
        response = client.chat.completions.parse(
            model=model,
            messages=messages,  # type: ignore
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            extra_body=extra_body,
        )
        return json.loads(response.choices[0].message.content)  # type: ignore
    except Exception as e:
        raise RuntimeError(f"error in generation:\n{e}") from e


def _is_rate_limit_error(e: BaseException) -> bool:
    """
    Determine if an exception is (or was caused by) a rate limit error.

    Checks the whole exception chain, because the generate helpers wrap API
    errors in a RuntimeError. A 429 status counts regardless of message text
    (e.g. OpenRouter's "429 - Provider returned error" when upstream is limited).
    :param e: The exception
    :return: True if this is a rate limit error
    """
    seen: set[int] = set()
    cur: BaseException | None = e
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        # openai.APIStatusError (incl. RateLimitError) carries `status_code`
        if getattr(cur, "status_code", None) == 429:
            return True
        if "rate limit" in str(cur).lower().replace("_", " "):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


class _FakeStatusError(Exception):
    """Stand-in for openai.APIStatusError (which needs a live response object)."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_is_rate_limit_error() -> None:
    api_error = _FakeStatusError(
        "Error code: 429 - Provider returned error", 429
    )

    # direct, and wrapped as the generate helpers do
    assert _is_rate_limit_error(api_error)
    try:
        try:
            raise api_error
        except Exception as inner:
            raise RuntimeError(f"error in generation:\n{inner}") from inner
    except RuntimeError as wrapped:
        assert _is_rate_limit_error(wrapped)

    # message-based fallback still works
    assert _is_rate_limit_error(RuntimeError("Rate_limit exceeded"))

    # other errors are not retried
    assert not _is_rate_limit_error(
        _FakeStatusError("Error code: 404 - No endpoints found", 404)
    )
    assert not _is_rate_limit_error(ValueError("bad output"))


def test_with_retry_retries_wrapped_429() -> None:
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            err = _FakeStatusError(
                "Error code: 429 - Provider returned error", 429
            )
            raise RuntimeError("error in generation") from err
        return "ok"

    sleeps: list[float] = []
    assert with_retry(flaky, sleep=sleeps.append)() == "ok"
    assert calls["n"] == 3
    assert sleeps == [1.0, 1.5]


def with_retry(
    func: Callable,
    max_retries: int = 20,
    initial_sleep_time: float = 1.0,
    backoff_factor: float = 1.5,
    sleep: Callable[[float | int], None] = time.sleep,
) -> Callable:
    """
    Retry a function with exponential backoff.

    This decorator retries the wrapped function in case of rate limit errors, using an exponential
    backoff strategy to increase the wait time between retries.

    :param func: The function to be retried.
    :param max_retries: Maximum number of retry attempts.
    :param initial_sleep_time: Initial sleep time in seconds.
    :param backoff_factor: Factor by which the sleep time increases after each retry.
    :return callable: A wrapped version of the input function with retry logic.
    """

    def wrapper(*args, **kwargs) -> Any:
        sleep_time = initial_sleep_time
        for _ in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if _is_rate_limit_error(e):
                    # sleep for the current period
                    sleep(sleep_time)
                    # increment by the backoff factor
                    sleep_time *= backoff_factor
                    # try again
                    continue

                # not a rate limit error; raise
                raise

        raise RuntimeError("maximum retries exceeded")

    return wrapper
