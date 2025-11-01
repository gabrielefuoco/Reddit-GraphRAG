"""Pydantic models that define the data contracts for the entire application.

These are the single source of truth for the data structures.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class PoliticalEntity(BaseModel):
    """A political entity mentioned in a text."""

    name: str = Field(description="Canonical name of the entity, e.g., 'Joe Biden'")
    type: str = Field(description="Type of entity (e.g., PERSON, ORG, GPE)")


class Stance(BaseModel):
    """Represents a stance towards a political entity."""

    target_entity_name: str = Field(
        description="Name of the target entity for the stance"
    )
    stance: Literal["FAVORABLE", "AGAINST", "NEUTRAL"] = Field(
        description="The detected stance"
    )
    confidence: float = Field(
        description="Model confidence score (softmax output)", ge=0.0, le=1.0
    )
    sentence: str = Field(description="The exact sentence where the stance was detected")


class Post(BaseModel):
    """Represents a Reddit post."""

    id: str = Field(description="The unique ID of the post.")
    author: Optional[str] = Field(
        default="deleted",
        description="The Reddit username of the post's author. 'deleted' if the author is unknown.",
    )
    content: str = Field(description="The raw, original content of the Reddit post.")
    cleaned_content: str = Field(
        description="The post's content after cleaning and preprocessing for NLP tasks."
    )
    timestamp: int = Field(description="UTC Unix timestamp of the post's creation.")
    score: int = Field(description="The score of the post (upvotes - downvotes).")
    subreddit: str = Field(
        description="The name of the subreddit where the post was published."
    )
    entities: List[PoliticalEntity] = Field(
        default_factory=list,
        description="A list of political entities identified in the post's content.",
    )
    stances: List[Stance] = Field(
        default_factory=list,
        description="A list of political stances detected within the post's content.",
    )
    embedding: List[float] = Field(
        default_factory=list,
        description="A high-dimensional vector embedding representing the semantic meaning of the post's content.",
    )

class Comment(BaseModel):
    """Represents a Reddit comment."""

    id: str = Field(description="The unique ID of the comment.")
    post_id: str = Field(description="The ID of the parent post.")
    author: Optional[str] = Field(
        default="deleted",
        description="The Reddit username of the comment's author.",
    )
    content: str = Field(description="The raw, original content of the Reddit comment.")
    cleaned_content: str = Field(
        description="The comment's content after cleaning for NLP tasks."
    )
    timestamp: int = Field(description="UTC Unix timestamp of the comment's creation.")
    score: int = Field(description="The score of the comment.")
    entities: List[PoliticalEntity] = Field(
        default_factory=list,
        description="A list of political entities identified in the comment's content.",
    )
    stances: List[Stance] = Field(
        default_factory=list,
        description="A list of political stances detected within the comment's content.",
    )
    embedding: List[float] = Field(
        default_factory=list,
        description="A high-dimensional vector embedding for the comment.",
    )