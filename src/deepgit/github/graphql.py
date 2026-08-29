from __future__ import annotations

import asyncio
import logging
import os

import httpx

from deepgit.config import get_settings
from deepgit.github.schema import RepoRecord, TreeEntry

logger = logging.getLogger(__name__)

GITHUB_GRAPHQL = "https://api.github.com/graphql"
CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = 30.0
MAX_RETRIES = 3

# Common README filenames tried in a single round-trip via aliased fields.
_README_ALIASES: dict[str, str] = {
    "readme_md": "HEAD:README.md",
    "readme_rst": "HEAD:README.rst",
    "readme_txt": "HEAD:README.txt",
    "readme_markdown": "HEAD:README.markdown",
    "readme_bare": "HEAD:README",
    "readme_lower": "HEAD:readme.md",
    "docs_index": "HEAD:docs/index.md",
}

_README_FRAGMENTS = "\n".join(
    f'{alias}: object(expression: "{expr}") {{ ... on Blob {{ text }} }}'
    for alias, expr in _README_ALIASES.items()
)

# NOTE: license spdxId (the old REST path had a `spi_id` typo — fixed here).
_SEARCH_QUERY = f"""
query($q: String!, $first: Int!, $after: String) {{
  rateLimit {{ cost remaining resetAt }}
  search(query: $q, type: REPOSITORY, first: $first, after: $after) {{
    repositoryCount
    pageInfo {{ endCursor hasNextPage }}
    nodes {{
      ... on Repository {{
        nameWithOwner
        name
        owner {{ login }}
        url
        description
        homepageUrl
        stargazerCount
        forkCount
        watchers {{ totalCount }}
        issues(states: OPEN) {{ totalCount }}
        primaryLanguage {{ name }}
        languages(first: 8, orderBy: {{ field: SIZE, direction: DESC }}) {{
          nodes {{ name }}
        }}
        repositoryTopics(first: 15) {{ nodes {{ topic {{ name }} }} }}
        licenseInfo {{ spdxId }}
        createdAt
        updatedAt
        pushedAt
        isArchived
        isFork
        diskUsage
        defaultBranchRef {{ name }}
        {_README_FRAGMENTS}
        rootTree: object(expression: "HEAD:") {{
          ... on Tree {{
            entries {{
              name
              type
              object {{ ... on Blob {{ byteSize }} }}
            }}
          }}
        }}
      }}
    }}
  }}
}}
"""


def _headers() -> dict[str, str]:
    token = get_settings().github_api_key or os.getenv("GITHUB_API_KEY", "")
    if not token:
        raise RuntimeError("GITHUB_API_KEY is required for the GraphQL client")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }


def _pick_readme(node: dict) -> str:
    for alias in _README_ALIASES:
        blob = node.get(alias)
        if blob and blob.get("text"):
            return blob["text"]
    return ""


def _parse_node(node: dict) -> RepoRecord | None:
    if not node or not node.get("nameWithOwner"):
        return None

    tree_entries: list[TreeEntry] = []
    root = node.get("rootTree") or {}
    for entry in root.get("entries", []) or []:
        obj = entry.get("object") or {}
        tree_entries.append(
            TreeEntry(
                name=entry.get("name", ""),
                type=entry.get("type", ""),
                size=int(obj.get("byteSize", 0) or 0),
                path=entry.get("name", ""),
            )
        )

    topics = [
        t["topic"]["name"]
        for t in (node.get("repositoryTopics", {}) or {}).get("nodes", [])
        if t.get("topic")
    ]
    languages = [
        n["name"]
        for n in (node.get("languages", {}) or {}).get("nodes", [])
        if n.get("name")
    ]

    return RepoRecord(
        name_with_owner=node["nameWithOwner"],
        name=node.get("name", ""),
        owner=(node.get("owner") or {}).get("login", ""),
        url=node.get("url", ""),
        description=node.get("description") or "",
        homepage=node.get("homepageUrl") or "",
        stars=int(node.get("stargazerCount", 0) or 0),
        forks=int(node.get("forkCount", 0) or 0),
        watchers=int((node.get("watchers") or {}).get("totalCount", 0) or 0),
        open_issues=int((node.get("issues") or {}).get("totalCount", 0) or 0),
        primary_language=(node.get("primaryLanguage") or {}).get("name", "") or "",
        languages=languages,
        topics=topics,
        license_spdx=(node.get("licenseInfo") or {}).get("spdxId", "") or "",
        created_at=node.get("createdAt", "") or "",
        updated_at=node.get("updatedAt", "") or "",
        pushed_at=node.get("pushedAt", "") or "",
        is_archived=bool(node.get("isArchived", False)),
        is_fork=bool(node.get("isFork", False)),
        default_branch=(node.get("defaultBranchRef") or {}).get("name", "main")
        or "main",
        disk_usage_kb=int(node.get("diskUsage", 0) or 0),
        readme=_pick_readme(node),
        root_tree=tree_entries,
    )


class GitHubGraphQL:
    """Thin async wrapper over the GitHub GraphQL endpoint."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> "GitHubGraphQL":
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers=_headers(),
                timeout=httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT),
            )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def _post(self, variables: dict) -> dict:
        assert self._client is not None
        payload = {"query": _SEARCH_QUERY, "variables": variables}
        for attempt in range(MAX_RETRIES):
            try:
                resp = await self._client.post(GITHUB_GRAPHQL, json=payload)
            except (httpx.ReadTimeout, httpx.ConnectTimeout):
                await asyncio.sleep(2**attempt)
                continue
            if resp.status_code == 200:
                # A 200 can still carry an empty or non-JSON body when GitHub
                # applies secondary rate limiting to rapid bursts of searches.
                # Treat that as a transient failure and retry rather than crash.
                try:
                    data = resp.json()
                except ValueError:
                    logger.warning("empty/non-JSON 200 body, retrying")
                    await asyncio.sleep(2**attempt)
                    continue
                if data.get("errors"):
                    logger.warning("GraphQL errors: %s", data["errors"])
                return data.get("data", {}) or {}
            if resp.status_code in (403, 429):
                wait = min(2**attempt * 5, 60)
                logger.warning("Rate limited (%s), waiting %ss", resp.status_code, wait)
                await asyncio.sleep(wait)
                continue
            if resp.status_code >= 500:
                await asyncio.sleep(2**attempt)
                continue
            logger.error("GraphQL POST -> %s: %s", resp.status_code, resp.text[:200])
            return {}
        return {}

    async def search(
        self,
        query: str,
        *,
        limit: int = 50,
        page_size: int = 50,
    ) -> list[RepoRecord]:
        """Search repositories, paginating until ``limit`` records are gathered.

        A single page returns full metadata + README + root tree for every
        repo — the whole point of the redesign.
        """
        records: list[RepoRecord] = []
        after: str | None = None
        page_size = min(page_size, 100)

        while len(records) < limit:
            want = min(page_size, limit - len(records))
            data = await self._post({"q": query, "first": want, "after": after})
            search = data.get("search") or {}
            rate = data.get("rateLimit") or {}
            if rate:
                logger.debug(
                    "GraphQL cost=%s remaining=%s", rate.get("cost"), rate.get("remaining")
                )
            for node in search.get("nodes", []) or []:
                rec = _parse_node(node)
                if rec is not None:
                    records.append(rec)
            page = search.get("pageInfo") or {}
            if not page.get("hasNextPage"):
                break
            after = page.get("endCursor")
            if not after:
                break

        return records[:limit]

    async def fetch_tree(
        self,
        name_with_owner: str,
        path: str = "",
        *,
        ref: str = "HEAD",
    ) -> list[TreeEntry]:
        """List the entries of one directory in a repo (one round-trip).

        Used by the adaptive controller to look one level deeper than the root
        tree so escalation reads the *real* source file (e.g. ``src/pkg/core.py``)
        instead of guessing paths. ``path`` is repo-root-relative ("" = root).
        """
        assert self._client is not None
        owner, _, name = name_with_owner.partition("/")
        if not name:
            return []
        expr = f"{ref}:{path}" if path else f"{ref}:"
        query = (
            "query($owner: String!, $name: String!) {\n"
            "  repository(owner: $owner, name: $name) {\n"
            f'    object(expression: "{expr}") {{\n'
            "      ... on Tree { entries { name type "
            "object { ... on Blob { byteSize } } } }\n"
            "    }\n  }\n}"
        )
        payload = {"query": query, "variables": {"owner": owner, "name": name}}
        for attempt in range(MAX_RETRIES):
            try:
                resp = await self._client.post(GITHUB_GRAPHQL, json=payload)
            except (httpx.ReadTimeout, httpx.ConnectTimeout):
                await asyncio.sleep(2**attempt)
                continue
            if resp.status_code == 200:
                obj = (
                    ((resp.json().get("data") or {}).get("repository") or {}).get(
                        "object"
                    )
                    or {}
                )
                out: list[TreeEntry] = []
                for entry in obj.get("entries", []) or []:
                    blob = entry.get("object") or {}
                    child = entry.get("name", "")
                    out.append(
                        TreeEntry(
                            name=child,
                            type=entry.get("type", ""),
                            size=int(blob.get("byteSize", 0) or 0),
                            path=f"{path}/{child}" if path else child,
                        )
                    )
                return out
            if resp.status_code in (403, 429):
                await asyncio.sleep(min(2**attempt * 5, 60))
                continue
            if resp.status_code >= 500:
                await asyncio.sleep(2**attempt)
                continue
            return []
        return []

    async def fetch_files(
        self,
        name_with_owner: str,
        paths: list[str],
        *,
        ref: str = "HEAD",
        max_bytes: int = 60_000,
    ) -> dict[str, str]:
        """Fetch the text of several files from one repo in a single request.

        Used by the evidence sub-agent to read *actual code* — tests, source,
        and manifests — instead of only the README. Returns a mapping of
        ``path -> text`` (missing/binary files are omitted).
        """
        assert self._client is not None
        owner, _, name = name_with_owner.partition("/")
        if not name:
            return {}

        aliases: dict[str, str] = {}
        field_lines: list[str] = []
        for i, path in enumerate(paths):
            alias = f"f{i}"
            aliases[alias] = path
            expr = f"{ref}:{path}"
            field_lines.append(
                f'{alias}: object(expression: "{expr}") {{ '
                f"... on Blob {{ text isBinary byteSize }} }}"
            )
        fields = "\n".join(field_lines)
        query = (
            f"query($owner: String!, $name: String!) {{\n"
            f"  repository(owner: $owner, name: $name) {{\n{fields}\n  }}\n}}"
        )
        payload = {"query": query, "variables": {"owner": owner, "name": name}}

        for attempt in range(MAX_RETRIES):
            try:
                resp = await self._client.post(GITHUB_GRAPHQL, json=payload)
            except (httpx.ReadTimeout, httpx.ConnectTimeout):
                await asyncio.sleep(2**attempt)
                continue
            if resp.status_code == 200:
                data = (resp.json().get("data") or {}).get("repository") or {}
                out: dict[str, str] = {}
                for alias, path in aliases.items():
                    blob = data.get(alias)
                    if not blob or blob.get("isBinary"):
                        continue
                    text = blob.get("text")
                    if text:
                        out[path] = text[:max_bytes]
                return out
            if resp.status_code in (403, 429):
                await asyncio.sleep(min(2**attempt * 5, 60))
                continue
            if resp.status_code >= 500:
                await asyncio.sleep(2**attempt)
                continue
            return {}
        return {}


async def search_repositories(query: str, *, limit: int = 50) -> list[RepoRecord]:
    """Convenience one-shot batched search."""
    async with GitHubGraphQL() as gh:
        return await gh.search(query, limit=limit)
