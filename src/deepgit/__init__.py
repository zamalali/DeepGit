from deepgit.pipeline import SearchResult, search, search_structured

__version__ = "4.0.0"


def deep_search(intent: str, **kwargs) -> str:
    """Agentic deep search that reads real repository code (opt-in escalation)."""
    from deepgit.agent import search as _agent_search

    return _agent_search(intent, **kwargs)


__all__ = ["search", "search_structured", "SearchResult", "deep_search", "__version__"]
