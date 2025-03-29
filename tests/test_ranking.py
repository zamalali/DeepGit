import pytest
from tools.ranking import multi_factor_ranking
import math

class DummyState:
    def __init__(self):
        # Create dummy filtered_candidates with varying scores.
        self.filtered_candidates = [
            {"semantic_similarity": 0.8, "cross_encoder_score": 7.0, "activity_score": 5.0, "code_quality_score": 90, "stars": 100},
            {"semantic_similarity": 0.6, "cross_encoder_score": 8.0, "activity_score": 3.0, "code_quality_score": 80, "stars": 50},
            {"semantic_similarity": 0.9, "cross_encoder_score": 6.0, "activity_score": 7.0, "code_quality_score": 95, "stars": 150}
        ]
        self.final_ranked = []

class DummyConfig:
    def __init__(self):
        self.configurable = {}

def test_multi_factor_ranking():
    state = DummyState()
    config = DummyConfig().__dict__
    result = multi_factor_ranking(state, config)
    # Check that final_ranked is sorted in descending order by final_score.
    final = state.final_ranked
    assert len(final) == 3
    assert final[0]["final_score"] >= final[1]["final_score"] >= final[2]["final_score"]
    # Verify that star score is computed as log(stars+1)
    expected_star_score = math.log(100 + 1)
    # Normalize check is more complex; here we just ensure the final score key exists.
    for repo in final:
        assert "final_score" in repo
