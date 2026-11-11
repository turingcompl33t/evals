"""
Shared models.
"""

from typing import Any, Literal

from pydantic import BaseModel, model_validator

# reasoning settings: "default" sends nothing (provider/model default applies)
ReasoningSetting = Literal["default", "on", "off"]


class Example(BaseModel):
    """An individual evaluation example."""

    # unique example ID
    id: int
    # the ID of the source MCQ (from mcq_raw.json)
    question_id: int
    # the ID of the prepended biography (from bios.jsonl)
    bio_id: int
    # the prompt / question
    question: str
    # the answer that matches the behavior
    answer_matching_behavior: Literal["a", "b"]
    # the answer that does not match behavior
    answer_not_matching_behavior: Literal["a", "b"]
    # the user's affiliation
    user_affiliation: Literal["conservative", "liberal"]


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
    # the requested reasoning setting
    reasoning: ReasoningSetting
    # the pinned OpenRouter provider; None if routing was left to OpenRouter
    provider: str | None
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
