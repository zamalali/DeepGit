import os
import logging
import getpass
from pathlib import Path
from dotenv import load_dotenv
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from typing import List, Any
from tools.llm_config import LLMConfig, LLMProvider

# ---------------------------
# Import node functions
# ---------------------------
from tools.convert_query import convert_searchable_query
from tools.parse_hardware import parse_hardware_spec
from tools.github import ingest_github_repos
from tools.dense_retrieval import hybrid_dense_retrieval
from tools.cross_encoder_reranking import cross_encoder_reranking
from tools.filtering import threshold_filtering
from tools.dependency_analysis import dependency_analysis
from tools.activity_analysis import repository_activity_analysis
from tools.decision_maker import decision_maker
from tools.code_quality import code_quality_analysis
from tools.merge_analysis import merge_analysis
from tools.ranking import multi_factor_ranking
from tools.output_presentation import output_presentation

# ---------------------------
# Logging & Environment Setup
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

dotenv_path = Path(__file__).resolve().parent.parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

if "GITHUB_API_KEY" not in os.environ:
    os.environ["GITHUB_API_KEY"] = getpass.getpass("Enter your GitHub API key: ")

# ---------------------------
# State & Configuration
# ---------------------------
@dataclass(kw_only=True)
class AgentState:
    user_query: str = field(default="")
    searchable_query: str = field(default="")
    hardware_spec: str = field(default="")               # extracted hardware hint
    repositories: List[Any] = field(default_factory=list)
    semantic_ranked: List[Any] = field(default_factory=list)
    reranked_candidates: List[Any] = field(default_factory=list)
    filtered_candidates: List[Any] = field(default_factory=list)
    hardware_filtered: List[Any] = field(default_factory=list)
    activity_candidates: List[Any] = field(default_factory=list)
    quality_candidates: List[Any] = field(default_factory=list)
    final_ranked: List[Any] = field(default_factory=list)
    llm_config: LLMConfig = field(default_factory=lambda: LLMConfig(provider=LLMProvider.OPENAI))

@dataclass(kw_only=True)
class AgentStateInput:
    user_query: str = field(default="")
    llm_config: LLMConfig = field(default_factory=lambda: LLMConfig(provider=LLMProvider.OPENAI))

@dataclass(kw_only=True)
class AgentStateOutput:
    final_results: str = field(default="")

class AgentConfiguration(BaseModel):
    max_results: int = Field(100, title="Max Results", description="Max GitHub results")
    per_page: int = Field(25, title="Per Page", description="GitHub results per page")
    dense_retrieval_k: int = Field(100, title="Dense K", description="Top‑K for dense retrieval")
    cross_encoder_top_n: int = Field(50, title="Cross‑encoder N", description="Top‑N after re‑rank")
    min_stars: int = Field(50, title="Min Stars", description="Minimum star count")
    cross_encoder_threshold: float = Field(5.5, title="CE Threshold", description="Cross‑encoder score cutoff")
    sem_model_name: str = Field("all-mpnet-base-v2", title="SentenceTransformer model")
    cross_encoder_model_name: str = Field("cross-encoder/ms-marco-MiniLM-L-6-v2", title="Cross‑encoder model")

    @classmethod
    def from_runnable_config(cls, config: Any = None) -> "AgentConfiguration":
        cfg = (config or {}).get("configurable", {})
        raw = {k: os.environ.get(k.upper(), cfg.get(k)) for k in cls.__fields__.keys()}
        values = {k: v for k, v in raw.items() if v is not None}
        return cls(**values)

# -------------------------------------------------------
# Build & Compile the Workflow Graph
# -------------------------------------------------------
builder = StateGraph(
    AgentState,
    input=AgentStateInput,
    output=AgentStateOutput,
    config_schema=AgentConfiguration
)

# Core nodes
builder.add_node("convert_searchable_query", convert_searchable_query)
builder.add_node("parse_hardware",         parse_hardware_spec)
builder.add_node("ingest_github_repos",    ingest_github_repos)
builder.add_node("neural_dense_retrieval", hybrid_dense_retrieval)
builder.add_node("cross_encoder_reranking",cross_encoder_reranking)
builder.add_node("threshold_filtering",    threshold_filtering)
builder.add_node("dependency_analysis",    dependency_analysis)
builder.add_node("repository_activity_analysis", repository_activity_analysis)
builder.add_node("decision_maker",         decision_maker)
builder.add_node("code_quality_analysis",  code_quality_analysis)
builder.add_node("merge_analysis",         merge_analysis)
builder.add_node("multi_factor_ranking",   multi_factor_ranking)
builder.add_node("output_presentation",    output_presentation)

# Edges (dataflow)
builder.add_edge(START,                     "convert_searchable_query")
builder.add_edge("convert_searchable_query","parse_hardware")
builder.add_edge("parse_hardware",          "ingest_github_repos")
builder.add_edge("ingest_github_repos",     "neural_dense_retrieval")
builder.add_edge("neural_dense_retrieval",  "cross_encoder_reranking")
builder.add_edge("cross_encoder_reranking", "threshold_filtering")

# **Parallel branches** after filtering:
builder.add_edge("threshold_filtering",     "dependency_analysis")
builder.add_edge("threshold_filtering",     "repository_activity_analysis")
builder.add_edge("threshold_filtering",     "decision_maker")

# Merge the outputs of the three parallel paths:
builder.add_edge("dependency_analysis",     "code_quality_analysis")
builder.add_edge("decision_maker",          "code_quality_analysis")
builder.add_edge("repository_activity_analysis", "merge_analysis")
builder.add_edge("code_quality_analysis",   "merge_analysis")

builder.add_edge("merge_analysis",          "multi_factor_ranking")
builder.add_edge("multi_factor_ranking",    "output_presentation")
builder.add_edge("output_presentation",     END)

graph = builder.compile()

# -------------------------------------------------------
# CLI entrypoint
# -------------------------------------------------------
if __name__ == "__main__":
    initial = AgentStateInput(
        user_query=(
            "I am looking for chain-of-thought prompting for reasoning models "
            "and I am GPU-poor, so I need something lightweight."
        ),
        llm_config=LLMConfig(provider=LLMProvider.OPENAI)
    )
    result = graph.invoke(initial)
    print(result["final_results"])
