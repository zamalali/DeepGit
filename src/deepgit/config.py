from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class DeepGitSettings(BaseSettings):
    """Master configuration — loaded once, reused everywhere."""

    # --- API keys ---
    github_api_key: str = Field("", alias="GITHUB_API_KEY")
    groq_api_key: str = Field("", alias="GROQ_API_KEY")
    langsmith_api_key: str = Field("", alias="LANGSMITH_API_KEY")

    # --- LLM ---
    llm_provider: Literal["groq", "openai", "anthropic", "vertex_ai"] = "groq"
    llm_model: str = "deepseek-r1-distill-llama-70b"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 8192

    # --- Vertex AI (used when llm_provider == "vertex_ai") ---
    vertex_project: str = Field("", alias="VERTEX_PROJECT")
    vertex_location: str = Field("us-central1", alias="VERTEX_LOCATION")

    # --- GitHub search ---
    max_results: int = 100
    per_page: int = 25
    search_concurrency: int = 8
    github_rate_limit_rpm: int = 30

    # --- Retrieval ---
    colbert_model: str = "colbert-ir/colbertv2.0"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    retrieval_alpha: float = 0.7  # dense vs sparse blend
    dense_top_k: int = 100
    rerank_top_n: int = 50
    min_stars: int = 50
    cross_encoder_threshold: float = 5.5

    # --- Analysis ---
    enable_code_quality: bool = False  # expensive — opt-in
    activity_analysis_concurrency: int = 10
    max_doc_chars: int = 50_000

    # --- Ranking weights ---
    w_cross_encoder: float = 0.30
    w_semantic: float = 0.20
    w_activity: float = 0.15
    w_quality: float = 0.15
    w_stars: float = 0.20

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 7860

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @classmethod
    def from_runnable_config(cls, config: Any = None) -> "DeepGitSettings":
        """Build from LangGraph RunnableConfig['configurable']."""
        cfg = (config or {}).get("configurable", {})
        overrides = {k: v for k, v in cfg.items() if v is not None}
        return cls(**overrides)


@lru_cache(maxsize=1)
def get_settings() -> DeepGitSettings:
    """Singleton accessor — import and call once per process."""
    return DeepGitSettings()
