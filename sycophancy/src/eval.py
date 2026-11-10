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
from pydantic import BaseModel, model_validator
from tqdm.contrib.concurrent import thread_map

from api import generate_structured, with_retry
from models import Example

# the political affiliations ontology
AFFILIATIONS: Final[set[str]] = {"conservative", "liberal"}
# default maximum workers
DEFAULT_MAX_WORKERS = 4
# default temperate
DEFAULT_TEMPERATURE = 0.0
# default max tokens
DEFAULT_MAX_TOKENS = 1024


class ModelResponse(BaseModel):
    """The model response format"""

    # the answer
    answer: Literal["a", "b"]


class Result(BaseModel):
    """A single eval result (one generation)"""

    # the example ID
    id: int
    # the ID of the source MCQ
    question_id: int
    # the ID of the prepended biography
    bio_id: int
    # the user's affiliation
    user_affiliation: Literal["conservative", "liberal"]
    # the model identifier
    model: str
    # the sampling temperature
    temperature: float
    # the model answer; None if the generation failed
    model_answer: Literal["a", "b"] | None
    # the answer matching behavior
    answer_matching_behavior: Literal["a", "b"]
    # the other answer
    answer_not_matching_behavior: Literal["a", "b"]
    # the raw (parsed JSON) model response, if one was received
    raw_response: dict[str, Any] | None = None
    # the error message; None if the generation succeeded
    error: str | None = None

    @model_validator(mode="after")
    def _answer_xor_error(self) -> "Result":
        if (self.model_answer is None) == (self.error is None):
            raise ValueError("exactly one of model_answer, error must be set")
        return self


def _parse_args() -> tuple[Path, Path, str, int, bool]:
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


def _output_path(output_dir: Path, model: str) -> Path:
    """
    Generate the output path based on model name.
    :param output_dir: The output directory
    :param model: The model name
    :return: The output path
    """
    return (output_dir / model.replace("-", "_")).with_suffix(".jsonl")


def test_output_path() -> None:
    assert _output_path(Path("/tmp/results"), "gpt-4o-mini") == Path(
        "/tmp/results/gpt_4o_mini.jsonl"
    )


def _eval(
    *,
    client: OpenAI,
    model: str,
    examples: list[Example],
    output_path: Path,
    max_workers: int,
) -> None:
    """
    Run the eval; writing results.
    :param client: The client
    :param model: The model
    :param examples: The loaded examples
    :param output_path: The output path
    :param max_workers: Max workers
    """
    # construct generation helper with retry
    _generate = with_retry(generate_structured)

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
                max_tokens=DEFAULT_MAX_TOKENS,
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
    input_path, output_dir, model, max_workers, verbose = _parse_args()
    logging.basicConfig(level=logging.INFO if verbose else logging.ERROR)

    if "OPENROUTER_API_KEY" not in os.environ:
        raise ValueError("must set OPENROUTER_API_KEY")

    if not input_path.is_file():
        raise ValueError(f"missing input at {input_path}")
    output_dir.mkdir(exist_ok=True, parents=True)

    output_path = _output_path(output_dir, model)
    if output_path.is_file():
        raise ValueError(f"output path {output_path} already exists")

    logging.info(f"using input {input_path}")
    logging.info(f"using output {output_path}")
    logging.info(f"using model {model}")
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
        examples=examples,
        output_path=output_path,
        max_workers=max_workers,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
