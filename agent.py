import os
import base64
import requests
import numpy as np
import datetime
import math
import logging
import getpass
import faiss
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from typing import List, Any
import subprocess
import tempfile
import shutil
import stat

# Import node functions from the tools directory.
from tools.convert_query import convert_searchable_query
from tools.github import ingest_github_repos
from tools.dense_retrieval import hybrid_dense_retrieval
from tools.cross_encoder_reranking import cross_encoder_reranking
from tools.filtering import threshold_filtering
from tools.activity_analysis import repository_activity_analysis
from tools.decision_maker import decision_maker
from tools.code_quality import code_quality_analysis
from tools.merge_analysis import merge_analysis
from tools.ranking import multi_factor_ranking
from tools.output_presentation import output_presentation

# ---------------------------
# Logging and Environment Setup
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

dotenv_path = Path(__file__).resolve().parent.parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

if "GITHUB_API_KEY" not in os.environ:
    os.environ["GITHUB_API_KEY"] = getpass.getpass("Enter your GitHub API key: ")

# ---------------------------
# State and Configuration
# ---------------------------
@dataclass(kw_only=True)
class AgentState:
    user_query: str = field(default="I am researching the application of Chain of Thought prompting for improving reasoning in large language models within a Python environment.")
    searchable_query: str = field(default="")
    repositories: List[Any] = field(default_factory=list)
    semantic_ranked: List[Any] = field(default_factory=list)
    reranked_candidates: List[Any] = field(default_factory=list)
    filtered_candidates: List[Any] = field(default_factory=list)
    activity_candidates: List[Any] = field(default_factory=list)
    quality_candidates: List[Any] = field(default_factory=list)
    final_ranked: List[Any] = field(default_factory=list)
    run_code_analysis: bool = field(default=False)

@dataclass(kw_only=True)
class AgentStateInput:
    user_query: str = field(default="I am researching the application of Chain of Thought prompting for improving reasoning in large language models within a Python environment.")

"""
@dataclass(kw_only=True)
class AgentStateOutput:
    final_ranked: List[Any] = field(default_factory=list)
"""

@dataclass(kw_only=True)
class AgentStateOutput:
    final_results: str = ""

class AgentConfiguration(BaseModel):
    max_results: int = Field(default=100, title="Max Results", description="Maximum results to fetch from GitHub")
    per_page: int = Field(default=25, title="Per Page", description="Results per page for GitHub API")
    dense_retrieval_k: int = Field(default=100, title="Dense Retrieval Top K", description="Top K candidates to retrieve from FAISS")
    cross_encoder_top_n: int = Field(default=50, title="Cross Encoder Top N", description="Top N candidates after re-ranking")
    min_stars: int = Field(default=50, title="Minimum Stars", description="Minimum star count threshold for filtering")
    cross_encoder_threshold: float = Field(default=5.5, title="Cross Encoder Threshold", description="Threshold for cross encoder score filtering")
    
    sem_model_name: str = Field(default="all-mpnet-base-v2", title="Sentence Transformer Model", description="Model for dense retrieval")
    cross_encoder_model_name: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", title="Cross Encoder Model", description="Model for re-ranking")
    
    @classmethod
    def from_runnable_config(cls, config: Any = None) -> "AgentConfiguration":
        configurable = config["configurable"] if config and "configurable" in config else {}
        raw_values = {name: os.environ.get(name.upper(), configurable.get(name)) for name in cls.__fields__.keys()}
        values = {k: v for k, v in raw_values.items() if v is not None}
        return cls(**values)

# -------------------------------------------------------
# Build and Compile the Graph
# -------------------------------------------------------
builder = StateGraph(
    AgentState,
    input=AgentStateInput,
    output=AgentStateOutput,
    config_schema=AgentConfiguration
)

builder.add_node("convert_searchable_query", convert_searchable_query)
builder.add_node("ingest_github_repos", ingest_github_repos)
builder.add_node("neural_dense_retrieval", hybrid_dense_retrieval)
builder.add_node("cross_encoder_reranking", cross_encoder_reranking)
builder.add_node("threshold_filtering", threshold_filtering)
builder.add_node("repository_activity_analysis", repository_activity_analysis)
builder.add_node("decision_maker", decision_maker)
builder.add_node("code_quality_analysis", code_quality_analysis)
builder.add_node("merge_analysis", merge_analysis)
builder.add_node("multi_factor_ranking", multi_factor_ranking)
builder.add_node("output_presentation", output_presentation)

builder.add_edge(START, "convert_searchable_query")
builder.add_edge("convert_searchable_query", "ingest_github_repos")
builder.add_edge("ingest_github_repos", "neural_dense_retrieval")
builder.add_edge("neural_dense_retrieval", "cross_encoder_reranking")
builder.add_edge("cross_encoder_reranking", "threshold_filtering")
builder.add_edge("threshold_filtering", "repository_activity_analysis")
builder.add_edge("threshold_filtering", "decision_maker")
builder.add_edge("decision_maker", "code_quality_analysis")
builder.add_edge("repository_activity_analysis", "merge_analysis")
builder.add_edge("code_quality_analysis", "merge_analysis")
builder.add_edge("merge_analysis", "multi_factor_ranking")
builder.add_edge("multi_factor_ranking", "output_presentation")
builder.add_edge("output_presentation", END)

graph = builder.compile()

if __name__ == "__main__":
    initial_state = AgentStateInput(
        user_query="I am researching the application of Chain of Thought prompting for improving reasoning in large language models within a Python environment. No need for code analysis."
    )
    result = graph.invoke(initial_state)
    print(result["final_results"])

# -------------------------------------------------------