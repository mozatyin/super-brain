"""Core data models for the Super Brain personality system."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """Supporting quote for a detected trait."""
    text: str
    source: str
    timestamp: Optional[str] = None


class Trait(BaseModel):
    """A single personality trait measurement on a 0.0-1.0 scale."""
    dimension: str          # e.g., "OPN", "CON", "DRK"
    name: str               # e.g., "fantasy", "narcissism"
    value: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(default_factory=list)


class TraitRelation(BaseModel):
    """Correlation between two traits."""
    source: str
    target: str
    correlation: float = Field(ge=-1.0, le=1.0)
    direction: str  # "positive", "negative", "conditional"


class SampleSummary(BaseModel):
    """Metadata about the text sample used for detection."""
    total_tokens: int = Field(ge=0)
    conversation_count: int = Field(ge=0)
    date_range: list[str]
    contexts: list[str]
    confidence_overall: float = Field(ge=0.0, le=1.0)


class PersonalityDNA(BaseModel):
    """Complete personality profile for one person — 66 traits across 13 dimensions."""
    id: str
    version: str = "0.1"
    created: Optional[datetime] = None
    updated: Optional[datetime] = None

    sample_summary: SampleSummary
    traits: list[Trait] = Field(default_factory=list)
    trait_relations: list[TraitRelation] = Field(default_factory=list)
