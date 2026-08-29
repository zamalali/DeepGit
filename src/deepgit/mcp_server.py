from __future__ import annotations

import logging
import os

import anyio
from mcp.server.fastmcp import FastMCP

from deepgit import __version__
from deepgit.config import get_settings

logger = logging.getLogger("deepgit.mcp")

mcp = FastMCP(name="DeepGit")


def _require_github_token() -> None:
    token = get_settings().github_api_key or os.getenv("GITHUB_API_KEY", "")
    if not token:
        raise ValueError(
            "GITHUB_API_KEY is not set. DeepGit needs a GitHub token (a classic "
            "or fine-grained PAT with public-repo read access) to query the GitHub "
            "GraphQL API. Set it in the environment or in a .env file."
        )


def _repo_url(repo: str) -> str:
    repo = (repo or "").strip().strip("/")
    return f"https://github.com/{repo}" if repo else ""


def _llm_status() -> dict:
    """Best-effort check that the configured LLM provider has its credentials."""
    s = get_settings()
    provider = s.llm_provider
    ok = True
    hint = ""
    if provider == "vertex_ai":
        creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        if not creds:
            ok, hint = False, "Set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON."
        elif not os.path.exists(creds):
            ok, hint = False, f"GOOGLE_APPLICATION_CREDENTIALS points to a missing file: {creds}"
        elif not s.vertex_project:
            ok, hint = False, "Set VERTEX_PROJECT to your GCP project id."
    elif provider == "groq":
        if not (s.groq_api_key or os.getenv("GROQ_API_KEY")):
            ok, hint = False, "Set GROQ_API_KEY."
    elif provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            ok, hint = False, "Set OPENAI_API_KEY."
    elif provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            ok, hint = False, "Set ANTHROPIC_API_KEY."
    return {"provider": provider, "model": s.llm_model, "configured": ok, "hint": hint}


def _run_report(intent: str, limit: int) -> str:
    from deepgit.pipeline import _format_report, search_structured

    result = search_structured(intent, shortlist=max(8, limit))
    if not result.ranked and not result.candidates_scanned:
        return f"# DeepGit\n\nNo repositories found for: {intent}"
    return _format_report(intent, result.ranked[:limit], result.summary, result.trace)


def _run_json(intent: str, limit: int) -> dict:
    from deepgit.pipeline import search_structured

    result = search_structured(intent, shortlist=max(8, limit))
    return {
        "intent": result.intent,
        "summary": result.summary,
        "escalated": result.escalated,
        "candidates_scanned": result.candidates_scanned,
        "llm_calls": result.llm_calls,
        "elapsed_seconds": round(result.elapsed, 1),
        "results": [
            {
                "rank": i + 1,
                "repo": r.repo,
                "url": _repo_url(r.repo),
                "fit_score": round(r.fit_score, 3),
                "verdict": r.verdict,
                "why": r.why,
                "risks": list(r.risks),
            }
            for i, r in enumerate(result.ranked[:limit])
        ],
    }


@mcp.tool()
async def find_repositories(intent: str, limit: int = 5) -> str:
    """Find the best GitHub repositories for a described need and return a report.

    Give a natural-language description of what you are looking for — a use case,
    a capability, a stack, or a problem to solve — not just keywords. DeepGit
    expands the intent, searches GitHub across keyword, star-sorted topic, and
    semantic angles, judges the shortlist from README + project signals, and
    adaptively reads the real code of contested repos when the ranking is
    uncertain. Returns a ranked Markdown report with a fit verdict and rationale
    per repository.

    Examples of good ``intent`` values:
      - "a high-performance message broker for pub/sub and streaming in Go"
      - "type-safe form validation for React with good TypeScript support"
      - "library to parse and diff PDFs in Python"

    Args:
        intent: The developer's need, described in natural language.
        limit: How many top repositories to include in the report (1-10).
    """
    _require_github_token()
    limit = max(1, min(int(limit), 10))
    return await anyio.to_thread.run_sync(_run_report, intent, limit)


@mcp.tool()
async def find_repositories_json(intent: str, limit: int = 8) -> dict:
    """Find repositories for an intent and return structured JSON (for automation).

    Same engine as ``find_repositories`` but returns machine-readable results
    instead of a report: a ranked list with repo, URL, fit score, verdict,
    one-line rationale, and risks, plus run telemetry (whether DeepGit escalated
    to reading real code, how many repos it scanned, LLM-call count, and timing).
    Use this when you want to program against the results.

    Args:
        intent: The developer's need, described in natural language.
        limit: Maximum number of ranked repositories to return (1-20).
    """
    _require_github_token()
    limit = max(1, min(int(limit), 20))
    return await anyio.to_thread.run_sync(_run_json, intent, limit)


@mcp.tool()
async def deep_research(intent: str) -> str:
    """Run DeepGit's agentic deep-research lane over an intent (slower, deepest).

    This always uses the full agent: it plans, searches, reads real repository
    code (source, tests, manifests), and reflects before answering. Prefer this
    for high-stakes or ambiguous comparisons where you want maximum grounding and
    are willing to trade latency for depth. For everyday lookups, use
    ``find_repositories`` instead — it is fast and only reads code when uncertain.

    Args:
        intent: The developer's need, described in natural language.
    """
    _require_github_token()
    from deepgit import deep_search

    return await anyio.to_thread.run_sync(deep_search, intent)


@mcp.tool()
def check_setup() -> dict:
    """Verify DeepGit is configured correctly — call this first, it is instant.

    Returns whether the GitHub token and the LLM provider credentials are present,
    and whether the optional semantic-recall layer (LanceDB + fastembed) is
    installed, along with a short hint for anything missing. Use it right after
    wiring DeepGit into your client to confirm a search will actually work — it
    runs no search and costs nothing.
    """
    github_ok = bool(get_settings().github_api_key or os.getenv("GITHUB_API_KEY", ""))
    llm = _llm_status()

    semantic: dict = {"installed": False, "indexed_repos": 0}
    try:
        from deepgit.retrieval import get_index

        index = get_index()
        semantic["installed"] = bool(index.available)
        if index.available:
            semantic["indexed_repos"] = index.count()
    except Exception:  # pragma: no cover - optional deps
        pass

    ready = github_ok and llm["configured"]
    return {
        "deepgit_version": __version__,
        "ready": ready,
        "github_token": {
            "configured": github_ok,
            "hint": "" if github_ok else "Set GITHUB_API_KEY to a GitHub PAT with public-repo read access.",
        },
        "llm": llm,
        "semantic_recall": {
            **semantic,
            "hint": "" if semantic["installed"] else "Optional: pip install 'deepgit[semantic]' for cross-query recall.",
        },
        "message": (
            "DeepGit is ready — try find_repositories."
            if ready
            else "DeepGit is not fully configured; see the hints above."
        ),
    }


def main() -> None:
    logging.basicConfig(
        level=os.getenv("DEEPGIT_LOG_LEVEL", "INFO"),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("DeepGit MCP server v%s starting", __version__)
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport == "streamable-http":
        mcp.run(
            transport="streamable-http",
            host=os.environ.get("HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "8080")),
        )
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
