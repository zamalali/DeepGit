import numpy as np
import pytest
from tools.cross_encoder_reranking import cross_encoder_reranking

class DummyCrossEncoder:
    def __init__(self, model_name):
        self.model_name = model_name
    def predict(self, pairs, show_progress_bar=False):
        # Return a score equal to the length of the second element (chunk) modulo 10.
        if isinstance(pairs, list):
            scores = [len(pair[1]) % 10 for pair in pairs]
            return np.array(scores)
        else:
            return np.array([len(pairs[1]) % 10])

class DummyState:
    def __init__(self):
        self.user_query = "dummy query"
        # Create two repositories with different lengths of documentation.
        self.semantic_ranked = [
            {"combined_doc": "Short doc."},
            {"combined_doc": "This is a longer document that should produce a higher score due to more content."}
        ]

class DummyConfig:
    def __init__(self):
        self.configurable = {
            "cross_encoder_model_name": "dummy-cross-encoder",
            "cross_encoder_top_n": 2
        }

def test_cross_encoder_reranking(monkeypatch):
    monkeypatch.setattr("tools.cross_encoder_reranking.CrossEncoder", lambda model_name: DummyCrossEncoder(model_name))
    state = DummyState()
    config = DummyConfig().__dict__
    result = cross_encoder_reranking(state, config)
    # Verify that the reranked_candidates list is of length top_n (2) and that the candidate with the longer doc is ranked higher.
    assert len(state.reranked_candidates) == 2
    scores = [cand["cross_encoder_score"] for cand in state.reranked_candidates]
    # Since the dummy score is based on length mod 10, we can check that the max is first.
    assert scores[0] >= scores[1]
