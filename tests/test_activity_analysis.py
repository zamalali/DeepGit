import pytest
import datetime
from tools.activity_analysis import repository_activity_analysis

class DummyResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json = json_data
    def json(self):
        return self._json

def dummy_requests_get(url, headers=None, params=None):
    # Return dummy responses based on URL.
    if "pulls" in url:
        # Return 2 open pull requests.
        return DummyResponse(200, [{"dummy": "pr1"}, {"dummy": "pr2"}])
    elif "commits" in url:
        # Return a commit with a recent date.
        recent_date = (datetime.datetime.utcnow() - datetime.timedelta(days=10)).isoformat() + "Z"
        return DummyResponse(200, [{"commit": {"committer": {"date": recent_date}}}])
    return DummyResponse(200, {})

class DummyState:
    def __init__(self):
        self.filtered_candidates = [
            {"full_name": "dummy/repo1", "open_issues_count": 5},
            {"full_name": "dummy/repo2", "open_issues_count": 10}
        ]

class DummyConfig:
    def __init__(self):
        self.configurable = {}

def test_repository_activity_analysis(monkeypatch):
    monkeypatch.setattr("tools.activity_analysis.requests.get", dummy_requests_get)
    state = DummyState()
    config = DummyConfig().__dict__
    result = repository_activity_analysis(state, config)
    # Each candidate should now have an "activity_score" key.
    for repo in state.activity_candidates:
        assert "activity_score" in repo
        # Since the dummy returns 2 PRs and a commit 10 days ago, score should be computed.
        assert isinstance(repo["activity_score"], float)
