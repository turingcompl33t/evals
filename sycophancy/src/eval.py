"""
Eval implementation.
"""

import argparse
import logging
import os
import sys
import threading
from pathlib import Path
from queue import Queue
from typing import Any, Final, Literal

from openai import OpenAI
from pydantic import BaseModel
from tqdm.contrib.concurrent import thread_map

from api import generate_structured, with_retry
from models import Example, Result

# the political affiliations ontology
AFFILIATIONS: Final[set[str]] = {"conservative", "liberal"}
# default maximum workers
DEFAULT_MAX_WORKERS = 4
# default temperate
DEFAULT_TEMPERATURE = 0.0
# default max tokens (includes reasoning tokens, when reasoning is on)
DEFAULT_MAX_TOKENS = 1024

# reasoning settings: "default" sends nothing (provider/model default applies)
ReasoningSetting = Literal["default", "on", "off"]
# the OpenRouter `reasoning` request field for each explicit setting
REASONING_CONFIG: Final[dict[str, dict[str, Any]]] = {
    "on": {"enabled": True},
    "off": {"effort": "none"},
}


class ModelResponse(BaseModel):
    """The model response format"""

    # the answer
    answer: Literal["a", "b"]


def _parse_args() -> (
    tuple[Path, Path, str, str | None, ReasoningSetting, int, int, bool]
):
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="The path to the raw MCQ input",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="The directory to which results are written; naming is automatically applied",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="The model identifier"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Pin requests to this OpenRouter provider slug (no fallbacks; must support all request parameters)",
    )
    parser.add_argument(
        "--reasoning",
        choices=["default", "on", "off"],
        default="default",
        help="Reasoning setting; 'default' leaves it to the model/provider (default: default)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum output tokens, including reasoning tokens (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"The maximum number of workers for concurrent operations (default: {DEFAULT_MAX_WORKERS})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    args = parser.parse_args()

    return (
        args.input,
        args.output,
        args.model,
        args.provider,
        args.reasoning,
        args.max_tokens,
        args.max_workers,
        args.verbose,
    )


def _load_examples(path: Path) -> list[Example]:
    """
    Load examples from file.
    :param path: The path
    :return: The loaded examples
    """
    assert path.is_file(), "broken precondition"
    with path.open("r") as f:
        return [Example.model_validate_json(line.strip()) for line in f]


def _sanitize(name: str) -> str:
    """Make a model / provider identifier safe for use in a filename."""
    return (
        name.replace("/", "__")
        .replace(":", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def _output_path(
    output_dir: Path,
    model: str,
    provider: str | None = None,
    reasoning: ReasoningSetting = "default",
) -> Path:
    """
    Generate the output path based on run configuration.
    :param output_dir: The output directory
    :param model: The model name
    :param provider: The pinned provider, if any
    :param reasoning: The reasoning setting
    :return: The output path
    """
    name = _sanitize(model)
    if provider is not None:
        name += f"__{_sanitize(provider)}"
    if reasoning != "default":
        name += f"__reasoning_{reasoning}"
    return output_dir / f"{name}.jsonl"


def test_output_path() -> None:
    assert _output_path(Path("/tmp/results"), "gpt-4o-mini") == Path(
        "/tmp/results/gpt_4o_mini.jsonl"
    )
    # slashes, dots and colons stay in a single flat filename
    assert _output_path(
        Path("/tmp/results"), "meta-llama/llama-3.1-8b-instruct:free"
    ) == Path("/tmp/results/meta_llama__llama_3_1_8b_instruct_free.jsonl")
    # provider and reasoning distinguish runs of the same model
    assert _output_path(
        Path("/tmp/results"), "qwen/qwen3.5-9b", "deepinfra/fp8", "off"
    ) == Path(
        "/tmp/results/qwen__qwen3_5_9b__deepinfra__fp8__reasoning_off.jsonl"
    )


def _extra_body(
    provider: str | None, reasoning: ReasoningSetting
) -> dict[str, Any] | None:
    """
    Construct the OpenRouter-specific request body fields.
    :param provider: The provider to pin, if any
    :param reasoning: The reasoning setting
    :return: The extra body, or None if empty
    """
    body: dict[str, Any] = {}
    if provider is not None:
        body["provider"] = {
            "only": [provider],
            "allow_fallbacks": False,
            # otherwise a provider may silently ignore e.g. `reasoning`
            "require_parameters": True,
        }
    if reasoning != "default":
        body["reasoning"] = REASONING_CONFIG[reasoning]
    return body or None


def test_extra_body() -> None:
    assert _extra_body(None, "default") is None
    assert _extra_body(None, "off") == {"reasoning": {"effort": "none"}}
    assert _extra_body("together", "on") == {
        "provider": {
            "only": ["together"],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
        "reasoning": {"enabled": True},
    }


def _eval(
    *,
    client: OpenAI,
    model: str,
    provider: str | None,
    reasoning: ReasoningSetting,
    max_tokens: int,
    examples: list[Example],
    output_path: Path,
    max_workers: int,
) -> None:
    """
    Run the eval; writing results.
    :param client: The client
    :param model: The model
    :param provider: The provider to pin, if any
    :param reasoning: The reasoning setting
    :param max_tokens: Maximum output tokens
    :param examples: The loaded examples
    :param output_path: The output path
    :param max_workers: Max workers
    """
    # construct generation helper with retry
    _generate = with_retry(generate_structured)
    extra_body = _extra_body(provider, reasoning)

    # queue writes
    write_queue: Queue[Result] = Queue()

    def writer() -> None:
        with output_path.open("w") as f:
            for _ in range(len(examples)):
                res: Result = write_queue.get()
                f.write(f"{res.model_dump_json()}\n")
                f.flush()

    def generate_wrapper(ex: Example) -> None:
        def make_result(
            model_answer: Literal["a", "b"] | None,
            raw_response: dict[str, Any] | None,
            error: str | None = None,
        ) -> Result:
            return Result(
                id=ex.id,
                question_id=ex.question_id,
                bio_id=ex.bio_id,
                user_affiliation=ex.user_affiliation,
                model=model,
                temperature=DEFAULT_TEMPERATURE,
                reasoning=reasoning,
                provider=provider,
                model_answer=model_answer,
                answer_matching_behavior=ex.answer_matching_behavior,
                answer_not_matching_behavior=ex.answer_not_matching_behavior,
                raw_response=raw_response,
                error=error,
            )

        response: dict[str, Any] | None = None
        try:
            response = _generate(
                client=client,
                model=model,
                messages=[{"role": "user", "content": ex.question}],
                response_format=ModelResponse,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=max_tokens,
                extra_body=extra_body,
            )
            parsed = ModelResponse.model_validate(response)
            result = make_result(parsed.answer, response)
        except Exception as e:
            # request or parse failed; record the failure as a row
            logging.warning(f"example {ex.id} failed: {e}")
            result = make_result(None, response, f"{type(e).__name__}: {e}")
        # always enqueue exactly one result per example so the writer terminates
        write_queue.put(result)

    writer_thread = threading.Thread(target=writer)
    writer_thread.start()

    # kick off the workers
    _ = thread_map(generate_wrapper, examples, max_workers=max_workers)

    # block on the writer thread
    writer_thread.join()


def main() -> int:
    (
        input_path,
        output_dir,
        model,
        provider,
        reasoning,
        max_tokens,
        max_workers,
        verbose,
    ) = _parse_args()
    logging.basicConfig(level=logging.INFO if verbose else logging.ERROR)

    if "OPENROUTER_API_KEY" not in os.environ:
        raise ValueError("must set OPENROUTER_API_KEY")

    if not input_path.is_file():
        raise ValueError(f"missing input at {input_path}")
    output_dir.mkdir(exist_ok=True, parents=True)

    output_path = _output_path(output_dir, model, provider, reasoning)
    if output_path.is_file():
        raise ValueError(f"output path {output_path} already exists")

    logging.info(f"using input {input_path}")
    logging.info(f"using output {output_path}")
    logging.info(f"using model {model}")
    logging.info(f"using provider {provider or '(OpenRouter routing)'}")
    logging.info(f"using reasoning={reasoning}")
    logging.info(f"using max_tokens={max_tokens}")
    logging.info(f"using max_workers={max_workers}")

    examples = _load_examples(input_path)
    logging.info(f"loaded {len(examples)} examples")

    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    _eval(
        client=client,
        model=model,
        provider=provider,
        reasoning=reasoning,
        max_tokens=max_tokens,
        examples=examples,
        output_path=output_path,
        max_workers=max_workers,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
