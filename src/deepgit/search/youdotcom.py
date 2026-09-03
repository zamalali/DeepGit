"""Optional You.com web-search gather lane.

DeepGit's keyword and star-sorted topic queries both hit the GitHub search
index, so they share its blind spot: repos that keyword search buries, or
whose canonical descriptions omit the terms a developer would actually use.
A general web search sees those repos where they live in discourse — awesome
lists, "X vs Y" comparisons, blog posts — and can hand their ``owner/name``
slugs back to the normal pipeline.

This lane is **opt-in**: it only runs when ``YDC_API_KEY`` is set, adds no new
dependencies (httpx is already used for the GitHub client), and degrades to a
no-op on any failure — the same contract as the optional semantic-recall
layer. Slugs surfaced by the web search are re-fetched through the GitHub
GraphQL client so every candidate downstream is judged from the same rich
evidence (metadata, README, root tree) as the other gather angles.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

import httpx

from deepgit.config import get_settings
from deepgit.github.schema import RepoRecord

logger = logging.getLogger(__name__)

YOU_SEARCH_ENDPOINT = "https://ydc-index.io/v1/search"
CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 15.0

# github.com/<owner>/<repo> links, tolerating extra path/query/fragment.
_GH_URL_RE = re.compile(
    r"github\.com/([A-Za-z\d](?:[A-Za-z0-9-]{0,37}[A-Za-z\d])?)"
    r"/([A-Za-z0-9_.-]+)",
    re.IGNORECASE,
)

# GitHub path segments that are site pages, never repos.
_NON_REPO_OWNERS = {
    "about",
    "features",
    "pricing",
    "security",
    "enterprise",
    "customer-stories",
    "readme",
    "topics",
    "collections",
    "events",
    "sponsors",
    "marketplace",
    "organizations",
    "orgs",
    "settings",
    "apps",
    "explore",
    "pulse",
    "login",
    "join",
    "signup",
    "personal",
    "site",
    "github",
    "contact",
    "services",
}


def _clean_repo_segment(seg: str) -> str:
    seg = seg.strip().strip("/").removesuffix(".git").rstrip(".")
    return seg


def extract_repo_slugs(results: list[dict[str, Any]]) -> list[str]:
    """Pull unique ``owner/name`` slugs out of You.com web results.

    Scans URLs first (highest signal), then titles/descriptions/snippets for
    inline github.com links. Reserved GitHub path segments are skipped.
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in results:
        texts = [item.get("url", "") or ""]
        for key in ("title", "description"):
            texts.append(item.get(key, "") or "")
        for sn in item.get("snippets", []) or []:
            texts.append(sn or "")
        for text in texts:
            for owner, repo in _GH_URL_RE.findall(text):
                owner_l = owner.lower()
                repo_clean = _clean_repo_segment(repo)
                if owner_l in _NON_REPO_OWNERS or not repo_clean:
                    continue
                slug = f"{owner}/{repo_clean}"
                if slug.lower() not in seen:
                    seen.add(slug.lower())
                    out.append(slug)
    return out


def youcom_enabled() -> bool:
    """True when the web-search lane should run (key present, not disabled)."""
    s = get_settings()
    if s.youcom_search is False:  # explicit opt-out
        return False
    return bool(s.youcom_api_key or os.getenv("YDC_API_KEY", ""))


def _api_key() -> str:
    s = get_settings()
    return s.youcom_api_key or os.getenv("YDC_API_KEY", "")


async def search_web(queries: list[str], *, count: int = 8) -> list[dict[str, Any]]:
    """Query the You.com Search API; return raw web result dicts.

    Failures (missing key, HTTP errors, timeouts) return ``[]`` — the lane
    never breaks the search pipeline.
    """
    key = _api_key()
    if not key or not queries:
        return []
    out: list[dict[str, Any]] = []
    timeout = httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT)
    async with httpx.AsyncClient(
        headers={"X-API-Key": key, "Content-Type": "application/json"},
        timeout=timeout,
    ) as client:
        for query in queries:
            try:
                resp = await client.post(YOU_SEARCH_ENDPOINT, json={"query": query, "count": count})
                resp.raise_for_status()
                data = resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                logger.warning("you.com web search failed for %r: %s", query, exc)
                continue
            web = (data.get("results") or {}).get("web") or []
            out.extend(r for r in web if isinstance(r, dict))
    return out


async def gather_web_candidates(queries: list[str], *, max_slugs: int = 10) -> list[str]:
    """Web-search gather step: unique repo slugs mentioned on the open web."""
    try:
        results = await search_web(queries)
    except Exception as exc:  # pragma: no cover - defensive, search_web catches HTTP
        logger.warning("you.com gather failed: %s", exc)
        return []
    slugs = extract_repo_slugs(results)
    logger.info("[you.com] web search surfaced %d repo slug(s)", len(slugs))
    return slugs[:max_slugs]


async def gather_web_records(queries: list[str], *, max_slugs: int = 10) -> list[RepoRecord]:
    """Full web-gather lane: search the web, then hydrate slugs via GraphQL.

    Returns rich :class:`RepoRecord` objects so the judge sees the same
    evidence (metadata + README + root tree) as for the other gather angles.
    Any failure along the way yields ``[]`` — never breaks the pipeline.
    """
    slugs = await gather_web_candidates(queries, max_slugs=max_slugs)
    if not slugs:
        return []
    try:
        return await fetch_records_for_slugs(slugs)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("you.com slug hydration failed: %s", exc)
        return []


async def fetch_records_for_slugs(slugs: list[str]) -> list[RepoRecord]:
    """Fetch full RepoRecords for web-surfaced slugs via the GitHub client.

    Uses the ``repo:owner/name`` search qualifier so each candidate comes
    back with the same metadata + README + root tree the other angles get.
    """
    from deepgit.github.graphql import GitHubGraphQL

    if not slugs:
        return []
    async with GitHubGraphQL() as gh:
        batches = await asyncio.gather(
            *(gh.search(f"repo:{slug}", limit=1) for slug in slugs),
            return_exceptions=True,
        )
    records: list[RepoRecord] = []
    for batch in batches:
        if isinstance(batch, BaseException):
            logger.warning("repo lookup failed: %s", batch)
            continue
        records.extend(batch)
    return records
