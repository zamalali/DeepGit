import pytest
from tools.convert_query import convert_searchable_query
from dataclasses import dataclass, field
from typing import List, Any

# Create a dummy AgentState and AgentStateInput for testing.
@dataclass
class DummyState:
    user_query: str = "Test query for convert"
    searchable_query: str = ""
    
@dataclass
class DummyConfig:
    configurable: dict = field(default_factory=lambda: {})

def test_convert_searchable_query():
    state = DummyState()
    config = DummyConfig().__dict__
    result = convert_searchable_query(state, config)
    # Expect the searchable_query to be non-empty and contain a colon.
    assert ":" in state.searchable_query
    assert "searchable_query" in result
