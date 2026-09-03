"""Offline unit tests for the optional You.com web-search gather lane.

These exercise the pure functions (slug extraction, gating logic) without any
network access or credentials.
"""

from __future__ import annotations

import pytest

from deepgit.config import DeepGitSettings
from deepgit.search.youdotcom import (
    extract_repo_slugs,
    gather_web_candidates,
    search_web,
    youcom_enabled,
)


class TestExtractRepoSlugs:
    def test_urls_are_the_highest_signal(self):
        results = [
            {"url": "https://github.com/lancedb/lancedb", "title": "", "snippets": []},
            {"url": "https://github.com/qdrant/qdrant/issues/1", "title": "", "snippets": []},
        ]
        assert extract_repo_slugs(results) == ["lancedb/lancedb", "qdrant/qdrant"]

    def test_slugs_from_titles_and_snippets(self):
        results = [
            {
                "url": "https://someblog.io/vector-dbs-compared",
                "title": "Vector DBs compared",
                "description": "See https://github.com/weaviate/weaviate for details.",
                "snippets": ["Also check github.com/milvus-io/milvus."],
            }
        ]
        slugs = extract_repo_slugs(results)
        assert "weaviate/weaviate" in slugs
        assert "milvus-io/milvus" in slugs

    def test_github_site_pages_are_skipped(self):
        results = [
            {"url": "https://github.com/about", "title": "", "snippets": []},
            {"url": "https://github.com/features", "title": "", "snippets": []},
            {"url": "https://github.com/collections", "title": "", "snippets": []},
        ]
        assert extract_repo_slugs(results) == []

    def test_git_suffix_and_dotgit_are_stripped(self):
        results = [
            {"url": "https://github.com/psf/requests.git", "title": "", "snippets": []},
            {"url": "https://github.com/aio-libs/aiohttp/", "title": "", "snippets": []},
        ]
        assert extract_repo_slugs(results) == ["psf/requests", "aio-libs/aiohttp"]

    def test_duplicates_are_deduplicated_case_insensitively(self):
        results = [
            {"url": "https://github.com/psf/requests", "title": "", "snippets": []},
            {"url": "https://github.com/PSF/Requests", "title": "", "snippets": []},
        ]
        assert extract_repo_slugs(results) == ["psf/requests"]

    def test_empty_and_malformed_results_are_tolerated(self):
        assert extract_repo_slugs([]) == []
        assert extract_repo_slugs([{}, {"url": None}, {"url": "https://example.com"}]) == []


class TestGating:
    def _settings(self, monkeypatch, **env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setattr(
            "deepgit.search.youdotcom.get_settings",
            lambda: DeepGitSettings(_env_file=None),
        )

    def test_disabled_without_a_key(self, monkeypatch):
        monkeypatch.delenv("YDC_API_KEY", raising=False)
        monkeypatch.delenv("DEEPGIT_YOUCOM_SEARCH", raising=False)
        self._settings(monkeypatch)
        assert youcom_enabled() is False

    def test_enabled_with_a_key(self, monkeypatch):
        self._settings(monkeypatch, YDC_API_KEY="test-key")
        monkeypatch.delenv("DEEPGIT_YOUCOM_SEARCH", raising=False)
        assert youcom_enabled() is True

    def test_hard_opt_out_beats_the_key(self, monkeypatch):
        self._settings(monkeypatch, YDC_API_KEY="test-key", DEEPGIT_YOUCOM_SEARCH="0")
        assert youcom_enabled() is False


class TestFailuresAreSilent:
    def _settings(self, monkeypatch, **env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setattr(
            "deepgit.search.youdotcom.get_settings",
            lambda: DeepGitSettings(_env_file=None),
        )

    async def test_search_web_without_key_returns_empty(self, monkeypatch):
        monkeypatch.delenv("YDC_API_KEY", raising=False)
        self._settings(monkeypatch)
        assert await search_web(["anything"]) == []

    async def test_gather_web_candidates_swallows_network_errors(self, monkeypatch):
        async def boom(*a, **k):
            raise RuntimeError("network down")

        self._settings(monkeypatch)
        monkeypatch.setattr("deepgit.search.youdotcom.search_web", boom)
        assert await gather_web_candidates(["anything"]) == []

    async def test_gather_web_candidates_respects_max_slugs(self, monkeypatch):
        async def fake_search(queries, **k):
            return [{"url": f"https://github.com/o{i}/r{i}"} for i in range(20)]

        self._settings(monkeypatch)
        monkeypatch.setattr("deepgit.search.youdotcom.search_web", fake_search)
        slugs = await gather_web_candidates(["q"], max_slugs=5)
        assert len(slugs) == 5


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("requests", "requests"),
        ("requests.git", "requests"),
        ("aiohttp/", "aiohttp"),
        ("some.repo.", "some.repo"),
    ],
)
def test_clean_repo_segment(raw, expected):
    from deepgit.search.youdotcom import _clean_repo_segment

    assert _clean_repo_segment(raw) == expected
