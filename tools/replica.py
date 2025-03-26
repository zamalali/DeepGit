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

# Import the LLM-based query conversion function from your chat module.
from tools.chat import convert_to_search_tags

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
    # Only the raw user query is provided.
    user_query: str = field(default="I am researching the application of Chain of Thought prompting for improving reasoning in large language models within a Python environment.")
    searchable_query: str = field(default="")  # Holds colon-separated keywords.
    repositories: List[Any] = field(default_factory=list)
    semantic_ranked: List[Any] = field(default_factory=list)
    reranked_candidates: List[Any] = field(default_factory=list)
    filtered_candidates: List[Any] = field(default_factory=list)
    # New fields for parallel branch outputs:
    activity_candidates: List[Any] = field(default_factory=list)
    quality_candidates: List[Any] = field(default_factory=list)
    final_ranked: List[Any] = field(default_factory=list)

@dataclass(kw_only=True)
class AgentStateInput:
    user_query: str = field(default="I am researching the application of Chain of Thought prompting for improving reasoning in large language models within a Python environment.")

@dataclass(kw_only=True)
class AgentStateOutput:
    final_ranked: List[Any] = field(default_factory=list)

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

# ----------------------------------------------------
# Node A: Convert Raw Query to Searchable Keywords
# ----------------------------------------------------
def convert_searchable_query(state: AgentState, config: Any):
    searchable = convert_to_search_tags(state.user_query)
    state.searchable_query = searchable
    logger.info(f"Converted searchable query: {searchable}")
    return {"searchable_query": searchable}

# ----------------------------------------------------
# Node 1: Ingest GitHub Repositories
# ----------------------------------------------------
def ingest_github_repos(state: AgentState, config: Any):
    headers = {
        "Authorization": f"token {os.getenv('GITHUB_API_KEY')}",
        "Accept": "application/vnd.github.v3+json"
    }
    # Helper functions for content retrieval:
    def fetch_readme_content(repo_full_name):
        readme_url = f"https://api.github.com/repos/{repo_full_name}/readme"
        response = requests.get(readme_url, headers=headers)
        if response.status_code == 200:
            readme_data = response.json()
            return base64.b64decode(readme_data['content']).decode('utf-8')
        return ""
    
    def fetch_file_content(download_url):
        try:
            response = requests.get(download_url)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            logger.error(f"Error fetching file: {e}")
        return ""
    
    def fetch_directory_markdown(repo_full_name, path):
        md_content = ""
        url = f"https://api.github.com/repos/{repo_full_name}/contents/{path}"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            items = response.json()
            for item in items:
                if item["type"] == "file" and item["name"].lower().endswith(".md"):
                    content = fetch_file_content(item["download_url"])
                    md_content += f"\n\n# {item['name']}\n" + content
        return md_content
    
    def fetch_repo_documentation(repo_full_name):
        doc_text = ""
        readme = fetch_readme_content(repo_full_name)
        if readme:
            doc_text += "# README\n" + readme
        root_url = f"https://api.github.com/repos/{repo_full_name}/contents"
        response = requests.get(root_url, headers=headers)
        if response.status_code == 200:
            items = response.json()
            for item in items:
                if item["type"] == "file" and item["name"].lower().endswith(".md"):
                    if item["name"].lower() != "readme.md":
                        content = fetch_file_content(item["download_url"])
                        doc_text += f"\n\n# {item['name']}\n" + content
                elif item["type"] == "dir" and item["name"].lower() in ["docs", "documentation"]:
                    doc_text += f"\n\n# {item['name']} folder\n" + fetch_directory_markdown(repo_full_name, item["name"])
        return doc_text if doc_text.strip() else "No documentation available."
    
    def fetch_github_repositories(query, max_results, per_page):
        url = "https://api.github.com/search/repositories"
        repositories = []
        num_pages = max_results // per_page
        for page in range(1, num_pages + 1):
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": per_page,
                "page": page
            }
            response = requests.get(url, headers=headers, params=params)
            if response.status_code != 200:
                logger.error(f"Error {response.status_code}: {response.json().get('message')}")
                break
            items = response.json().get('items', [])
            if not items:
                break
            for repo in items:
                repo_link = repo['html_url']
                full_name = repo.get('full_name', '')
                clone_url = repo.get('clone_url', f"https://github.com/{full_name}.git")
                doc_content = fetch_repo_documentation(full_name)
                star_count = repo.get('stargazers_count', 0)
                repositories.append({
                    "title": repo.get('name', 'No title available'),
                    "link": repo_link,
                    "clone_url": clone_url,
                    "combined_doc": doc_content,
                    "stars": star_count,
                    "full_name": full_name,
                    "open_issues_count": repo.get('open_issues_count', 0)
                })
        logger.info(f"Fetched {len(repositories)} repositories for query '{query}'.")
        return repositories

    keyword_list = [kw.strip() for kw in state.searchable_query.split(":") if kw.strip()]
    logger.info(f"Searchable keywords: {keyword_list}")
    all_repos = []
    for keyword in keyword_list:
        query = f"{keyword} language:python"
        repos = fetch_github_repositories(query,
                                          AgentConfiguration.from_runnable_config(config).max_results,
                                          AgentConfiguration.from_runnable_config(config).per_page)
        all_repos.extend(repos)
    seen = set()
    unique_repos = []
    for repo in all_repos:
        if repo["full_name"] not in seen:
            seen.add(repo["full_name"])
            unique_repos.append(repo)
    state.repositories = unique_repos
    logger.info(f"Total unique repositories fetched: {len(state.repositories)}")
    return {"repositories": state.repositories}

# ---------------------------------------------------------
# Node 2: Neural Dense Retrieval with FAISS
# ---------------------------------------------------------
def neural_dense_retrieval(state: AgentState, config: Any):
    agent_config = AgentConfiguration.from_runnable_config(config)
    sem_model = SentenceTransformer(agent_config.sem_model_name)
    
    docs = [repo.get("combined_doc", "") for repo in state.repositories]
    if not docs:
        logger.warning("No documents found. Skipping dense retrieval.")
        state.semantic_ranked = []
        return {"semantic_ranked": state.semantic_ranked}
    logger.info(f"Encoding {len(docs)} documents for dense retrieval...")
    doc_embeddings = sem_model.encode(docs, convert_to_numpy=True, show_progress_bar=True, batch_size=16)
    if doc_embeddings.ndim == 1:
        doc_embeddings = doc_embeddings.reshape(1, -1)
    def normalize_embeddings(embeddings):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-10)
    doc_embeddings = normalize_embeddings(doc_embeddings)
    query_embedding = sem_model.encode(state.user_query, convert_to_numpy=True)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    query_embedding = normalize_embeddings(query_embedding)[0]
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings)
    k = min(int(agent_config.dense_retrieval_k), doc_embeddings.shape[0])
    D, I = index.search(np.expand_dims(query_embedding, axis=0), k)
    for idx, score in zip(I[0], D[0]):
        state.repositories[idx]["semantic_similarity"] = score
    state.semantic_ranked = sorted(state.repositories, key=lambda x: x.get("semantic_similarity", 0), reverse=True)
    logger.info(f"Dense retrieval complete: {len(state.semantic_ranked)} candidates ranked.")
    return {"semantic_ranked": state.semantic_ranked}

# --------------------------------------------------------
# Node 3: Cross-Encoder Re-Ranking
# --------------------------------------------------------
def cross_encoder_reranking(state: AgentState, config: Any):
    agent_config = AgentConfiguration.from_runnable_config(config)
    cross_encoder = CrossEncoder(agent_config.cross_encoder_model_name)
    candidates_for_rerank = state.semantic_ranked[:100]
    logger.info(f"Re-ranking {len(candidates_for_rerank)} candidates with cross-encoder...")
    def cross_encoder_rerank_func(query, candidates, top_n):
        pairs = [[query, candidate["combined_doc"]] for candidate in candidates]
        scores = cross_encoder.predict(pairs, show_progress_bar=True)
        for candidate, score in zip(candidates, scores):
            candidate["cross_encoder_score"] = score
        return sorted(candidates, key=lambda x: x["cross_encoder_score"], reverse=True)[:top_n]
    state.reranked_candidates = cross_encoder_rerank_func(
        state.user_query,
        candidates_for_rerank,
        int(agent_config.cross_encoder_top_n)
    )
    logger.info(f"Cross-encoder re-ranking complete: {len(state.reranked_candidates)} candidates remain.")
    return {"reranked_candidates": state.reranked_candidates}

# ------------------------------------------------------------
# Node 4: Threshold-Based Filtering
# ------------------------------------------------------------
def threshold_filtering(state: AgentState, config: Any):
    agent_config = AgentConfiguration.from_runnable_config(config)
    filtered = []
    for repo in state.reranked_candidates:
        if repo["stars"] < agent_config.min_stars and repo.get("cross_encoder_score", 0) < agent_config.cross_encoder_threshold:
            continue
        filtered.append(repo)
    if not filtered:
        filtered = state.reranked_candidates
    state.filtered_candidates = filtered
    logger.info(f"Filtering complete: {len(state.filtered_candidates)} candidates remain.")
    return {"filtered_candidates": state.filtered_candidates}

# -------------------------------------------------------------
# Node 5A: Repository Activity Analysis
# -------------------------------------------------------------
def repository_activity_analysis(state: AgentState, config: Any):
    headers = {
        "Authorization": f"token {os.getenv('GITHUB_API_KEY')}",
        "Accept": "application/vnd.github.v3+json"
    }
    def analyze_repository_activity(repo):
        full_name = repo.get("full_name")
        pr_url = f"https://api.github.com/repos/{full_name}/pulls"
        pr_params = {"state": "open", "per_page": 100}
        pr_response = requests.get(pr_url, headers=headers, params=pr_params)
        pr_count = len(pr_response.json()) if pr_response.status_code == 200 else 0

        commits_url = f"https://api.github.com/repos/{full_name}/commits"
        commits_params = {"per_page": 1}
        commits_response = requests.get(commits_url, headers=headers, params=commits_params)
        if commits_response.status_code == 200:
            commit_data = commits_response.json()
            if commit_data:
                commit_date_str = commit_data[0]["commit"]["committer"]["date"]
                commit_date = datetime.datetime.fromisoformat(commit_date_str.rstrip("Z"))
                days_diff = (datetime.datetime.utcnow() - commit_date).days
            else:
                days_diff = 999
        else:
            days_diff = 999

        open_issues = repo.get("open_issues_count", 0)
        non_pr_issues = max(0, open_issues - pr_count)
        activity_score = (3 * pr_count) + non_pr_issues - (days_diff / 30)
        return {"pr_count": pr_count, "latest_commit_days": days_diff, "activity_score": activity_score}
    
    activity_list = []
    for repo in state.filtered_candidates:
        data = analyze_repository_activity(repo)
        repo.update(data)
        activity_list.append(repo)
    state.activity_candidates = activity_list
    logger.info("Repository activity analysis complete.")
    return {"activity_candidates": state.activity_candidates}

# -------------------------------------------------------------
# Node 5B: Code Quality Analysis
# -------------------------------------------------------------
def remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def analyze_code_quality(repo_info):
    full_name = repo_info.get('full_name', 'unknown')
    clone_url = repo_info.get('clone_url')
    if not clone_url:
        repo_info["code_quality_score"] = 0
        repo_info["code_quality_issues"] = 0
        repo_info["python_files"] = 0
        return repo_info
    temp_dir = tempfile.mkdtemp()
    repo_path = os.path.join(temp_dir, full_name.split("/")[-1])
    try:
        from git import Repo
        Repo.clone_from(clone_url, repo_path, depth=1, no_single_branch=True)
        py_files = list(Path(repo_path).rglob("*.py"))
        total_files = len(py_files)
        if total_files == 0:
            logger.info(f"No Python files found in {full_name}.")
            repo_info["code_quality_score"] = 0
            repo_info["code_quality_issues"] = 0
            repo_info["python_files"] = 0
            return repo_info
        process = subprocess.run(
            ["flake8", "--max-line-length=120", repo_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        output = process.stdout.strip()
        error_count = len(output.splitlines()) if output else 0
        issues_per_file = error_count / total_files
        if issues_per_file <= 2:
            score = 95 + (2 - issues_per_file) * 2.5
        elif issues_per_file <= 5:
            score = 70 + (5 - issues_per_file) * 6.5
        elif issues_per_file <= 10:
            score = 40 + (10 - issues_per_file) * 3
        else:
            score = max(10, 40 - (issues_per_file - 10) * 2)
        repo_info["code_quality_score"] = round(score)
        repo_info["code_quality_issues"] = error_count
        repo_info["python_files"] = total_files
        return repo_info
    except Exception as e:
        logger.error(f"Error analyzing {full_name}: {e}.")
        repo_info["code_quality_score"] = 0
        repo_info["code_quality_issues"] = 0
        repo_info["python_files"] = 0
        return repo_info
    finally:
        try:
            shutil.rmtree(temp_dir, onerror=remove_readonly)
        except Exception as cleanup_e:
            logger.error(f"Cleanup error for {full_name}: {cleanup_e}")

def code_quality_analysis(state: AgentState, config: Any):
    quality_list = []
    for repo in state.filtered_candidates:
        if "clone_url" not in repo:
            repo["clone_url"] = f"https://github.com/{repo['full_name']}.git"
        updated_repo = analyze_code_quality(repo)
        quality_list.append(updated_repo)
    state.quality_candidates = quality_list
    logger.info("Code quality analysis complete.")
    return {"quality_candidates": state.quality_candidates}

# -------------------------------------------------------------
# Node 5C: Merge Analysis Results
# -------------------------------------------------------------
def merge_analysis(state: AgentState, config: Any):
    merged = {}
    # Merge activity_candidates and quality_candidates by full_name.
    for repo in state.activity_candidates:
        merged[repo["full_name"]] = repo.copy()
    for repo in state.quality_candidates:
        if repo["full_name"] in merged:
            merged[repo["full_name"]].update(repo)
        else:
            merged[repo["full_name"]] = repo.copy()
    merged_list = list(merged.values())
    state.filtered_candidates = merged_list
    logger.info(f"Merged analysis results: {len(merged_list)} candidates.")
    return {"filtered_candidates": merged_list}

# -------------------------------------------------------
# Node 6: Multi-Factor Ranking
# -------------------------------------------------------
def multi_factor_ranking(state: AgentState, config: Any):
    # Retrieve scores: semantic, cross-encoder, activity, code quality, and stars.
    semantic_scores = [repo.get("semantic_similarity", 0) for repo in state.filtered_candidates]
    cross_encoder_scores = [repo.get("cross_encoder_score", 0) for repo in state.filtered_candidates]
    activity_scores = [repo.get("activity_score", -100) for repo in state.filtered_candidates]
    quality_scores = [repo.get("code_quality_score", 0) for repo in state.filtered_candidates]
    star_scores = [math.log(repo.get("stars", 0) + 1) for repo in state.filtered_candidates]

    min_sem, max_sem = min(semantic_scores), max(semantic_scores)
    min_ce, max_ce = min(cross_encoder_scores), max(cross_encoder_scores)
    min_act, max_act = min(activity_scores), max(activity_scores)
    min_quality, max_quality = min(quality_scores), max(quality_scores)
    min_star, max_star = min(star_scores), max(star_scores)

    def normalize(val, min_val, max_val):
        if max_val - min_val == 0:
            return 0.5
        return (val - min_val) / (max_val - min_val)

    for repo in state.filtered_candidates:
        norm_sem = normalize(repo.get("semantic_similarity", 0), min_sem, max_sem)
        norm_ce = normalize(repo.get("cross_encoder_score", 0), min_ce, max_ce)
        norm_act = normalize(repo.get("activity_score", -100), min_act, max_act)
        norm_quality = normalize(repo.get("code_quality_score", 0), min_quality, max_quality)
        norm_star = normalize(math.log(repo.get("stars", 0) + 1), min_star, max_star)
        # Weights: Cross-encoder 0.30, Semantic 0.20, Activity 0.15, Code Quality 0.15, Stars 0.20.
        repo["final_score"] = (0.30 * norm_ce +
                               0.20 * norm_sem +
                               0.15 * norm_act +
                               0.15 * norm_quality +
                               0.20 * norm_star)
    state.final_ranked = sorted(state.filtered_candidates, key=lambda x: x["final_score"], reverse=True)
    logger.info(f"Final multi-factor ranking computed for {len(state.final_ranked)} candidates.")
    return {"final_ranked": state.final_ranked}

# --------------------------------------------------------
# Node 7: Output Presentation
# --------------------------------------------------------
def output_presentation(state: AgentState, config: Any):
    results_str = "\n=== Final Ranked Repositories ===\n"
    top_n = 10
    for rank, repo in enumerate(state.final_ranked[:top_n], 1):
        results_str += f"\nFinal Rank: {rank}\n"
        results_str += f"Title: {repo['title']}\n"
        results_str += f"Link: {repo['link']}\n"
        results_str += f"Stars: {repo['stars']}\n"
        results_str += f"Semantic Similarity: {repo.get('semantic_similarity', 0):.4f}\n"
        results_str += f"Cross-Encoder Score: {repo.get('cross_encoder_score', 0):.4f}\n"
        results_str += f"Activity Score: {repo.get('activity_score', 0):.2f}\n"
        results_str += f"Code Quality Score: {repo.get('code_quality_score', 0)}\n"
        results_str += f"Final Score: {repo.get('final_score', 0):.4f}\n"
        results_str += f"Combined Doc Snippet: {repo['combined_doc'][:200]}...\n"
        results_str += '-' * 80 + "\n"
    return {"final_results": results_str}

# -------------------------------------------------------
# Build and Compile the Graph
# -------------------------------------------------------
builder = StateGraph(
    AgentState,
    input=AgentStateInput,
    output=AgentStateOutput,
    config_schema=AgentConfiguration
)

# Graph Nodes (order and branching):
builder.add_node("convert_searchable_query", convert_searchable_query)
builder.add_node("ingest_github_repos", ingest_github_repos)
builder.add_node("neural_dense_retrieval", neural_dense_retrieval)
builder.add_node("cross_encoder_reranking", cross_encoder_reranking)
builder.add_node("threshold_filtering", threshold_filtering)
# Branch from threshold filtering:
builder.add_node("repository_activity_analysis", repository_activity_analysis)
builder.add_node("code_quality_analysis", code_quality_analysis)
# Merge node to combine both branches:
builder.add_node("merge_analysis", merge_analysis)
builder.add_node("multi_factor_ranking", multi_factor_ranking)
builder.add_node("output_presentation", output_presentation)

# Build the graph edges:
builder.add_edge(START, "convert_searchable_query")
builder.add_edge("convert_searchable_query", "ingest_github_repos")
builder.add_edge("ingest_github_repos", "neural_dense_retrieval")
builder.add_edge("neural_dense_retrieval", "cross_encoder_reranking")
builder.add_edge("cross_encoder_reranking", "threshold_filtering")
# Branch out from threshold filtering into two nodes:
builder.add_edge("threshold_filtering", "repository_activity_analysis")
builder.add_edge("threshold_filtering", "code_quality_analysis")
# Merge the results from the two branches:
builder.add_edge("repository_activity_analysis", "merge_analysis")
builder.add_edge("code_quality_analysis", "merge_analysis")
builder.add_edge("merge_analysis", "multi_factor_ranking")
builder.add_edge("multi_factor_ranking", "output_presentation")
builder.add_edge("output_presentation", END)

graph = builder.compile()


if __name__ == "__main__":
    
    initial_state = AgentStateInput(
        user_query="I am researching the application of Chain of Thought prompting for improving reasoning in large language models within a Python environment."
    )
    result = graph.run(initial_state)
    print(result.final_ranked)
