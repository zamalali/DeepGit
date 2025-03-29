import pytest
from tools.filtering import threshold_filtering

class DummyState:
    def __init__(self):
        # Create dummy reranked candidates with stars and cross_encoder_score.
        self.reranked_candidates = [
            {"stars": 60, "cross_encoder_score": 6.0},
            {"stars": 30, "cross_encoder_score": 4.0},  # Should be filtered out if both criteria fail.
            {"stars": 80, "cross_encoder_score": 5.5}
        ]

class DummyConfig:
    def __init__(self):
        self.configurable = {
            "min_stars": 50,
            "cross_encoder_threshold": 5.5
        }

def test_threshold_filtering():
    state = DummyState()
    config = DummyConfig().__dict__
    result = threshold_filtering(state, config)
    # Expect that the candidate with 30 stars and 4.0 score is filtered out.
    filtered = state.filtered_candidates
    assert len(filtered) == 2
    for repo in filtered:
        assert repo["stars"] >= 50 or repo["cross_encoder_score"] >= 5.5
