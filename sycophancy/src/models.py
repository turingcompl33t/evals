"""
Shared models.
"""

from typing import Literal

from pydantic import BaseModel


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
