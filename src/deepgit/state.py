from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(kw_only=True)
class Repository:
    """Structured representation of a GitHub repository."""

    title: str = ""
    full_name: str = ""
    link: str = ""
    clone_url: str = ""
    description: str = ""
    combined_doc: str = ""
    stars: int = 0
    forks: int = 0
    open_issues_count: int = 0
    language: str = ""
    topics: list[str] = field(default_factory=list)
    license: str = ""
    created_at: str = ""
    updated_at: str = ""

    # Scores — populated progressively through the pipeline
    semantic_similarity: float = 0.0
    cross_encoder_score: float = 0.0
    activity_score: float = 0.0
    code_quality_score: float = 0.0
    final_score: float = 0.0

    # Activity metadata
    pr_count: int = 0
    commit_frequency_30d: int = 0
    latest_commit_days: int = 999
    code_quality_issues: int = 0
    python_files: int = 0

    # Hardware compatibility
    hardware_compatible: bool | None = None


@dataclass(kw_only=True)
class DeepGitState:
    """Full pipeline state flowing through the LangGraph workflow."""

    # Input
    user_query: str = ""

    # Query expansion
    search_tags: list[str] = field(default_factory=list)
    hardware_spec: str | None = None
    target_language: str = "python"

    # Pipeline stages
    repositories: list[Repository] = field(default_factory=list)
    semantic_ranked: list[Repository] = field(default_factory=list)
    reranked_candidates: list[Repository] = field(default_factory=list)
    filtered_candidates: list[Repository] = field(default_factory=list)
    analyzed_candidates: list[Repository] = field(default_factory=list)
    final_ranked: list[Repository] = field(default_factory=list)

    # Output
    final_results: str = ""
    result_cards: list[dict[str, Any]] = field(default_factory=list)

    # Control flags
    run_code_analysis: bool = False
    error: str | None = None


@dataclass(kw_only=True)
class DeepGitInput:
    """Minimal input schema for the public graph API."""

    user_query: str = ""


@dataclass(kw_only=True)
class DeepGitOutput:
    """Output schema — what the graph returns to callers."""

    final_results: str = ""
    result_cards: list[dict[str, Any]] = field(default_factory=list)
