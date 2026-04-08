# tools/llm_provider.py
"""
Multi-provider LLM factory for DeepGit.

Supports:
  - groq   (default) – via langchain_groq.ChatGroq
  - minimax         – via langchain_openai.ChatOpenAI (OpenAI-compatible API)

Configure via environment variables:
  LLM_PROVIDER        – "groq" (default) or "minimax"
  GROQ_API_KEY        – required when LLM_PROVIDER=groq
  MINIMAX_API_KEY     – required when LLM_PROVIDER=minimax
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env once at import time
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

# Provider → default model mapping
_DEFAULT_MODELS = {
    "groq": "deepseek-r1-distill-llama-70b",
    "minimax": "MiniMax-M2.7",
}

# MiniMax requires temperature in (0.0, 1.0]
_MINIMAX_BASE_URL = "https://api.minimax.io/v1"


def _detect_provider() -> str:
    """Auto-detect LLM provider from env vars when LLM_PROVIDER is not set."""
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    if os.getenv("MINIMAX_API_KEY"):
        return "minimax"
    return "groq"


def _clamp_temperature(temperature: float, provider: str) -> float:
    """Clamp temperature to valid range for the provider."""
    if provider == "minimax":
        # MiniMax requires (0.0, 1.0]
        return max(0.01, min(temperature, 1.0))
    return temperature


def create_llm(
    temperature: float = 0.3,
    max_tokens: int = 512,
    max_retries: int = 3,
    model: str | None = None,
):
    """
    Create a LangChain chat model for the configured provider.

    Parameters
    ----------
    temperature : float
        Sampling temperature (auto-clamped for MiniMax).
    max_tokens : int
        Maximum response tokens.
    max_retries : int
        Number of automatic retries on transient errors.
    model : str or None
        Model name override.  Falls back to the provider default.

    Returns
    -------
    langchain BaseChatModel instance
    """
    provider = _detect_provider()
    model = model or os.getenv("LLM_MODEL") or _DEFAULT_MODELS.get(provider)
    temperature = _clamp_temperature(temperature, provider)

    if provider == "minimax":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError(
                "MINIMAX_API_KEY must be set when LLM_PROVIDER=minimax"
            )
        logger.info(f"[LLM] Using MiniMax provider (model={model})")
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=_MINIMAX_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        logger.info(f"[LLM] Using Groq provider (model={model})")
        return ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. Supported: groq, minimax"
    )
