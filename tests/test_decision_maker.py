import pytest
from tools.decision_maker import decision_maker

# Create a dummy should_run_code_analysis function.
def dummy_should_run_code_analysis(query, repo_count):
    # Return 1 if repo_count is less than 50, else 0.
    return 1 if repo_count < 50 else 0

class DummyState:
    def __init__(self):
        self.user_query = "dummy query"
        self.filtered_candidates = [{}] * 30  # 30 repos.

class DummyConfig:
    def __init__(self):
        self.configurable = {}

def test_decision_maker(monkeypatch):
    monkeypatch.setattr("tools.decision_maker.should_run_code_analysis", dummy_should_run_code_analysis)
    state = DummyState()
    config = DummyConfig().__dict__
    result = decision_maker(state, config)
    # Since repo_count is 30 (<50), decision should be 1 => run_code_analysis True.
    assert state.run_code_analysis is True
