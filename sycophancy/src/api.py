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
) -> dict:
    """
    Generate a response using the OpenAI API.
    :param model: The name of the model to use (e.g., "gpt-4o-mini").
    :param messages: A list of message dictionaries with 'role' and 'content' keys.
    :param response_format: The response format
    :param temperature: Controls randomness in output. Higher values make output more random.
    :param max_tokens: The maximum number of tokens to generate.
    :return: The generated response from the model.
    """
    try:
        response = client.chat.completions.parse(
            model=model,
            messages=messages,  # type: ignore
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        return json.loads(response.choices[0].message.content)  # type: ignore
    except Exception as e:
        raise RuntimeError(f"error in generation:\n{e}") from e


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
                is_rate_limit_error = "rate limit" in str(e).lower().replace(
                    "_", " "
                )
                if is_rate_limit_error:
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
