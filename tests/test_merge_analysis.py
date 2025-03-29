import pytest
from tools.merge_analysis import merge_analysis

class DummyState:
    def __init__(self):
        self.activity_candidates = [
            {"full_name": "dummy/repo", "activity_score": 8.0},
            {"full_name": "dummy/repo2", "activity_score": 7.0}
        ]
        self.quality_candidates = [
            {"full_name": "dummy/repo", "code_quality_score": 90},
            {"full_name": "dummy/repo3", "code_quality_score": 80}
        ]
        self.filtered_candidates = []

class DummyConfig:
    def __init__(self):
        self.configurable = {}

def test_merge_analysis():
    state = DummyState()
    config = DummyConfig().__dict__
    result = merge_analysis(state, config)
    # Expect merged repos: dummy/repo, dummy/repo2, dummy/repo3.
    merged = state.filtered_candidates
    assert len(merged) == 3
    # Check that for dummy/repo, both scores are merged.
    for repo in merged:
        if repo["full_name"] == "dummy/repo":
            assert "activity_score" in repo
            assert "code_quality_score" in repo
