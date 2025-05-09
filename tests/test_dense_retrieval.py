import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from tools.dense_retrieval import hybrid_dense_retrieval
from tools.cross_encoder_reranking import cross_encoder_reranking

class MockOpenAI:
    def __init__(self):
        self.embeddings = MagicMock()
        self.chat = MagicMock()
        
    def create(self, model, input):
        # Return deterministic embeddings based on text length
        if isinstance(input, list):
            embeddings = [[len(t)] for t in input]
        else:
            embeddings = [[len(input)]]
        return MagicMock(data=[MagicMock(embedding=e) for e in embeddings])

    def completions(self, model, messages, temperature, max_tokens):
        # Return deterministic scores based on text length
        content = messages[1]["content"]
        doc_length = len(content.split("\n\nDocument:")[1].split("\n\nScore")[0])
        score = min(10, doc_length / 100)  # Simple scoring based on doc length
        return MagicMock(choices=[MagicMock(message=MagicMock(content=str(score)))])

@pytest.fixture
def mock_openai():
    with patch('openai.OpenAI') as mock:
        mock.return_value = MockOpenAI()
        yield mock

def test_hybrid_dense_retrieval(mock_openai):
    # Test implementation here
    pass

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
        self.bm25_ranked = [{"combined_doc": "Test document one.", "bm25_score": 0.5},
                           {"combined_doc": "Another test document with more text.", "bm25_score": 0.5}]
        self.semantic_ranked = [{"combined_doc": "Test document one."},
                              {"combined_doc": "Another test document with more text."}]

class DummyConfig:
    def __init__(self):
        self.configurable = {
            "embedding_model": "text-embedding-3-small",
            "reranking_model": "gpt-3.5-turbo",
            "cross_encoder_top_n": 10
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

def test_openai_reranking(mock_openai):
    state = DummyState()
    config = DummyConfig().__dict__
    result = cross_encoder_reranking(state, config)
    
    # Verify that reranking was performed
    assert len(state.reranked_candidates) <= len(state.semantic_ranked)
    assert all("cross_encoder_score" in candidate for candidate in state.reranked_candidates)
    
    # Verify scores are normalized between 0 and 1
    scores = [candidate["cross_encoder_score"] for candidate in state.reranked_candidates]
    assert all(0 <= score <= 1 for score in scores)
    
    # Verify sorting
    sorted_scores = sorted(scores, reverse=True)
    assert scores == sorted_scores
