from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from langchain_core.tools import tool

from deepgit.github.graphql import GitHubGraphQL
from deepgit.github.schema import RepoRecord
from deepgit.reasoning import FitJudge

logger = logging.getLogger("deepgit.tools")

# ---------------------------------------------------------------------------
# Per-process candidate cache (populated by github_search, read by others)
# ---------------------------------------------------------------------------
_CANDIDATES: dict[str, RepoRecord] = {}
_FILE_CACHE: dict[str, dict[str, str]] = {}
_judge = FitJudge()


def reset_candidates() -> None:
    """Clear the candidate cache (call at the start of each search)."""
    _CANDIDATES.clear()
    _FILE_CACHE.clear()


def get_candidate(name_with_owner: str) -> RepoRecord | None:
    return _CANDIDATES.get(name_with_owner)


def all_candidates() -> list[RepoRecord]:
    return list(_CANDIDATES.values())


def _run(coro: Any) -> Any:
    """Run an async coroutine from a sync tool body."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already inside an event loop (e.g. async agent) — use a fresh loop safely.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _summarize(rec: RepoRecord) -> dict[str, Any]:
    return {
        "repo": rec.name_with_owner,
        "stars": rec.stars,
        "language": rec.primary_language,
        "license": rec.license_spdx,
        "archived": rec.is_archived,
        "pushed_at": rec.pushed_at,
        "description": rec.description[:200],
        "topics": rec.topics[:8],
        "has_tests": rec.has_tests(),
        "has_ci": rec.has_ci(),
        "manifests": rec.manifest_files(),
    }


@tool
def github_search(query: str, limit: int = 30) -> str:
    """Search GitHub for repositories matching a query.

    Accepts full GitHub search syntax (e.g. "vector database language:rust
    stars:>500"). Returns a compact JSON list of candidates with metadata,
    license, activity, and whether they ship tests/CI/manifests. Full details
    (README, file tree) are cached for later read_repo_files / judge_repo_fit
    calls. Prefer several targeted queries over one broad query.
    """

    logger.info("[search] query=%r limit=%s", query, limit)

    async def _do() -> list[RepoRecord]:
        async with GitHubGraphQL() as gh:
            return await gh.search(query, limit=min(int(limit), 50))

    records = _run(_do())
    for rec in records:
        _CANDIDATES[rec.name_with_owner] = rec
    logger.info(
        "[search] -> %s repos (cache now %s): %s",
        len(records),
        len(_CANDIDATES),
        ", ".join(r.name_with_owner for r in records[:8]),
    )
    return json.dumps([_summarize(r) for r in records], ensure_ascii=False)


@tool
def list_repo_files(name_with_owner: str) -> str:
    """List the root file tree of a candidate repository.

    Use this to decide which source/test/manifest files are worth reading
    before judging fit. The repo must have appeared in a prior github_search.
    """
    logger.info("[list_files] %s", name_with_owner)
    rec = _CANDIDATES.get(name_with_owner)
    if rec is None:
        return json.dumps({"error": f"{name_with_owner} not in candidates. Search first."})
    entries = [
        {"name": e.name, "type": e.type, "size": e.size} for e in rec.root_tree
    ]
    return json.dumps(
        {
            "repo": name_with_owner,
            "default_branch": rec.default_branch,
            "entries": entries,
            "manifests": rec.manifest_files(),
        },
        ensure_ascii=False,
    )


@tool
def read_repo_files(name_with_owner: str, paths: list[str]) -> str:
    """Read the actual text of specific files from a repository.

    This is how DeepGit verifies capability from *real code* rather than the
    README. Pass repo-root-relative paths (e.g. ["pyproject.toml",
    "src/pkg/core.py", "tests/test_core.py"]). Returns each file's text
    (truncated). The repo must have appeared in a prior github_search.
    """
    rec = _CANDIDATES.get(name_with_owner)
    if rec is None:
        return json.dumps({"error": f"{name_with_owner} not in candidates. Search first."})

    logger.info("[read_files] %s <- %s", name_with_owner, paths)
    cache = _FILE_CACHE.setdefault(name_with_owner, {})
    wanted = [p for p in paths if p not in cache]

    if wanted:
        async def _do() -> dict[str, str]:
            async with GitHubGraphQL() as gh:
                return await gh.fetch_files(name_with_owner, wanted, ref=rec.default_branch)

        fetched = _run(_do())
        cache.update(fetched)
        logger.info("[read_files] fetched %s file(s)", len(fetched))

    result = {p: cache.get(p, "") for p in paths}
    return json.dumps({"repo": name_with_owner, "files": result}, ensure_ascii=False)


def _build_evidence(rec: RepoRecord) -> str:
    parts = [
        f"REPO: {rec.name_with_owner}",
        f"stars={rec.stars} forks={rec.forks} language={rec.primary_language} "
        f"license={rec.license_spdx} archived={rec.is_archived} pushed_at={rec.pushed_at}",
        f"topics: {', '.join(rec.topics[:10])}",
        f"tests={rec.has_tests()} ci={rec.has_ci()} manifests={rec.manifest_files()}",
        "",
        "DESCRIPTION:",
        rec.description or "(none)",
        "",
        "README (excerpt):",
        (rec.readme or "(none)")[:6000],
    ]
    files = _FILE_CACHE.get(rec.name_with_owner, {})
    if files:
        parts.append("\nCODE EXCERPTS:")
        for path, text in files.items():
            parts.append(f"\n--- {path} ---\n{text[:3000]}")
    return "\n".join(parts)


@tool
def judge_repo_fit(
    name_with_owner: str,
    intent: str,
    must_have: list[str] | None = None,
    anti_patterns: list[str] | None = None,
) -> str:
    """Judge how well a repository fits the intent, grounded in gathered evidence.

    Assembles metadata + README + any code you have already read via
    read_repo_files, then runs the compiled DSPy JudgeFit module. Returns a
    calibrated verdict (fit_score 0..1, verdict, satisfied/missing must-haves,
    reasons, risks). Read the key source/test files BEFORE judging for the most
    accurate verdict.
    """
    rec = _CANDIDATES.get(name_with_owner)
    if rec is None:
        return json.dumps({"error": f"{name_with_owner} not in candidates. Search first."})

    logger.info("[judge] %s ...", name_with_owner)
    evidence = _build_evidence(rec)
    verdict = _judge(
        intent=intent,
        evidence=evidence,
        must_have=must_have or [],
        anti_patterns=anti_patterns or [],
    )
    payload = verdict.model_dump()
    payload["repo"] = name_with_owner
    logger.info(
        "[judge] %s -> %s (fit=%.2f)",
        name_with_owner,
        payload.get("verdict"),
        payload.get("fit_score", 0.0),
    )
    return json.dumps(payload, ensure_ascii=False)


TOOLS = [github_search, list_repo_files, read_repo_files, judge_repo_fit]
