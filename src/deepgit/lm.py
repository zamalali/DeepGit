from __future__ import annotations

import logging

import dspy

from deepgit.config import get_settings

logger = logging.getLogger(__name__)


def configure_dspy() -> None:
    """Initialize DSPy with the configured LLM provider.

    Call this once at startup before any DSPy module is used.
    """
    settings = get_settings()

    lm_kwargs: dict = {
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
    }

    if settings.llm_provider == "groq":
        model_name = f"groq/{settings.llm_model}"
    elif settings.llm_provider == "openai":
        model_name = f"openai/{settings.llm_model}"
    elif settings.llm_provider == "anthropic":
        model_name = f"anthropic/{settings.llm_model}"
    elif settings.llm_provider == "vertex_ai":
        model_name = f"vertex_ai/{settings.llm_model}"
        if settings.vertex_project:
            lm_kwargs["vertex_project"] = settings.vertex_project
        if settings.vertex_location:
            lm_kwargs["vertex_location"] = settings.vertex_location
    else:
        model_name = settings.llm_model

    lm = dspy.LM(model=model_name, **lm_kwargs)

    dspy.configure(lm=lm)
    logger.info(f"DSPy configured with {model_name}")
