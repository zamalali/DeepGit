import pytest
from tools.output_presentation import output_presentation

class DummyState:
    def __init__(self):
        # Create dummy final_ranked list.
        self.final_ranked = [
            {"title": "Repo1", "link": "https://github.com/repo1", "stars": 100,
             "semantic_similarity": 0.8, "cross_encoder_score": 7.0, "activity_score": 5.0,
             "code_quality_score": 90, "final_score": 0.95, "combined_doc": "This is a documentation snippet for Repo1."},
            {"title": "Repo2", "link": "https://github.com/repo2", "stars": 50,
             "semantic_similarity": 0.6, "cross_encoder_score": 6.5, "activity_score": 3.0,
             "code_quality_score": 80, "final_score": 0.85, "combined_doc": "Documentation snippet for Repo2."}
        ]

class DummyConfig:
    def __init__(self):
        self.configurable = {}

def test_output_presentation():
    state = DummyState()
    config = DummyConfig().__dict__
    result = output_presentation(state, config)
    output_str = result["final_results"]
    # Check that the output string contains expected repository titles and snippets.
    assert "Repo1" in output_str
    assert "https://github.com/repo1" in output_str
    assert "Documentation snippet" in output_str
