import pytest
import asyncio
import base64
from tools.github import ingest_github_repos

# Dummy response class to simulate httpx responses.
class DummyResponse:
    def __init__(self, status_code, json_data, text_data="Dummy text"):
        self.status_code = status_code
        self._json = json_data
        self.text = text_data

    def json(self):
        return self._json

# Dummy async get function simulating httpx.AsyncClient.get.
async def dummy_get(url, headers=None, params=None):
    # For repository search:
    if "search/repositories" in url:
        return DummyResponse(200, {"items": [{
            "html_url": "https://github.com/dummy/repo",
            "full_name": "dummy/repo",
            "clone_url": "https://github.com/dummy/repo.git",
            "stargazers_count": 100,
            "open_issues_count": 5,
            "name": "repo"
        }]})
    # For contents endpoint:
    elif "contents" in url:
        if "readme" in url.lower():
            # Return a dummy README encoded in base64.
            encoded = base64.b64encode("Dummy README".encode("utf-8")).decode("utf-8")
            return DummyResponse(200, {"content": encoded})
        else:
            # Return a dummy markdown file list.
            return DummyResponse(200, [{"type": "file", "name": "README.md", "download_url": "https://dummy/readme"}])
    return DummyResponse(200, {})

# Dummy fetch_file_content function for asynchronous calls.
async def dummy_fetch_file_content(download_url, client):
    return "Dummy documentation content."

# Patch the httpx.AsyncClient.get and file content retrieval in the module.
@pytest.fixture(autouse=True)
def patch_httpx(monkeypatch):
    import httpx
    # Patch the get method of AsyncClient to use dummy_get.
    monkeypatch.setattr(httpx.AsyncClient, "get", lambda self, url, headers=None, params=None: dummy_get(url, headers, params))
    # Patch fetch_file_content function in the tools.github module.
    monkeypatch.setattr("tools.github.fetch_file_content", dummy_fetch_file_content)

# Define dummy state and configuration classes.
class DummyState:
    def __init__(self):
        self.searchable_query = "dummy:repo"
        self.repositories = []

class DummyConfig:
    def __init__(self):
        self.configurable = {}

# Since ingest_github_repos is a synchronous wrapper calling asyncio.run,
# we can test it directly.
def test_ingest_github_repos():
    state = DummyState()
    config = DummyConfig().__dict__
    result = ingest_github_repos(state, config)
    # We expect at least one repository with the required keys.
    assert len(state.repositories) >= 1
    repo = state.repositories[0]
    for key in ["title", "link", "clone_url", "combined_doc", "stars", "full_name", "open_issues_count"]:
        assert key in repo
