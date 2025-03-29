import numpy as np
import pytest
from tools.dense_retrieval import hybrid_dense_retrieval

class DummySentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name
    def encode(self, texts, convert_to_numpy=True, **kwargs):
        # Return a deterministic vector for each text (e.g., using length)
        if isinstance(texts, list):
            return np.array([[len(t)] for t in texts], dtype=float)
        else:
            return np.array([len(texts)], dtype=float)

class DummyState:
    def __init__(self):
        self.user_query = "dummy query"
        self.repositories = [{"combined_doc": "Test document one."},
                             {"combined_doc": "Another test document with more text."}]

class DummyConfig:
    def __init__(self):
        self.configurable = {
            "sem_model_name": "dummy-model",
            "dense_retrieval_k": 10
        }

def test_neural_dense_retrieval(monkeypatch):
    monkeypatch.setattr("tools.dense_retrieval.SentenceTransformer", lambda model_name: DummySentenceTransformer(model_name))
    state = DummyState()
    config = DummyConfig().__dict__
    result = hybrid_dense_retrieval(state, config)
    # Expect semantic_ranked to be sorted in descending order of embedding (here, length).
    ranked = state.semantic_ranked
    assert len(ranked) == len(state.repositories)
    assert ranked[0]["semantic_similarity"] >= ranked[-1]["semantic_similarity"]
