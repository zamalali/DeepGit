import os
import base64
import requests
import numpy as np
import datetime
import torch
import math
import logging
import getpass
from typing import Dict, List, Any, TypedDict
from pathlib import Path
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END


# ---------------------------
# Environment and .env Setup
# ---------------------------
# Resolve the path to the root directory's .env file
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=str(dotenv_path))
# ------------------------------------------------------------------
# Bitsandbytes & Environment Setup
# ------------------------------------------------------------------
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["BITSANDBYTES_DISABLE_GPU"] = "1"

# Load .env if available
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=str(dotenv_path))

if "GITHUB_API_KEY" not in os.environ:
    os.environ["GITHUB_API_KEY"] = getpass.getpass("Enter your GitHub API key: ")

# ------------------------------------------------------------------
# Logging Setup
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# ChatGroq Setup (for query enhancement and justification)
# ------------------------------------------------------------------
llm_groq = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=100,
    timeout=15,
    max_retries=2
)

# ------------------------------------------------------------------
# GitHub Headers Setup
# ------------------------------------------------------------------
gh_headers = {
    "Authorization": f"token {os.environ.get('GITHUB_API_KEY')}",
    "Accept": "application/vnd.github.v3+json"
}

# ------------------------------------------------------------------
# Define the Agent State
# ------------------------------------------------------------------
class AgentState(TypedDict):
    original_query: str
    enhanced_query: str
    github_query: str
    candidates: List[Dict[str, Any]]
    final_ranked: List[Dict[str, Any]]
    justifications: Dict[str, str]

# ------------------------------------------------------------------
# Helper Functions for Repository Documentation
# ------------------------------------------------------------------
def fetch_file_content(download_url: str) -> str:
    try:
        response = requests.get(download_url)
        if response.status_code == 200:
            return response.text
    except Exception as e:
        logger.error(f"Error fetching file from {download_url}: {e}")
    return ""

def fetch_directory_markdown(repo_full_name: str, path: str) -> str:
    md_content = ""
    url = f"https://api.github.com/repos/{repo_full_name}/contents/{path}"
    response = requests.get(url, headers=gh_headers)
    if response.status_code == 200:
        items = response.json()
        for item in items:
            if item["type"] == "file" and item["name"].lower().endswith(".md"):
                content = fetch_file_content(item["download_url"])
                md_content += f"\n\n# {item['name']}\n" + content
    return md_content

def fetch_repo_documentation(repo_full_name: str) -> str:
    doc_text = ""
    readme_url = f"https://api.github.com/repos/{repo_full_name}/readme"
    response = requests.get(readme_url, headers=gh_headers)
    if response.status_code == 200:
        readme_data = response.json()
        try:
            decoded = base64.b64decode(readme_data['content']).decode('utf-8')
        except Exception as e:
            decoded = ""
            logger.error(f"Error decoding readme for {repo_full_name}: {e}")
        doc_text += "# README\n" + decoded
    root_url = f"https://api.github.com/repos/{repo_full_name}/contents"
    response = requests.get(root_url, headers=gh_headers)
    if response.status_code == 200:
        items = response.json()
        for item in items:
            if item["type"] == "file" and item["name"].lower().endswith(".md") and item["name"].lower() != "readme.md":
                content = fetch_file_content(item["download_url"])
                doc_text += f"\n\n# {item['name']}\n" + content
            elif item["type"] == "dir" and item["name"].lower() in ["docs", "documentation"]:
                doc_text += f"\n\n# {item['name']} folder\n" + fetch_directory_markdown(repo_full_name, item["name"])
    return doc_text if doc_text.strip() else "No documentation available."

# ------------------------------------------------------------------
# Helper: Extract Enhanced Query as a String
# ------------------------------------------------------------------
def get_enhanced_query(original_query: str) -> str:
    result = enhance_query_tool.invoke({"original_query": original_query})
    # If the result has a 'content' attribute, extract it; otherwise convert to string.
    return result.content if hasattr(result, "content") else str(result)

# ------------------------------------------------------------------
# Tool Definitions for Each Stage of the Pipeline
# ------------------------------------------------------------------

@tool
def enhance_query_tool(original_query: str) -> str:
    """
    Enhances the query for GitHub search by adding technical keywords and context,
    then returns only a valid GitHub search query using GitHub search syntax.
    """
    prompt = f"""You are an expert GitHub search assistant. Given the research topic: "{original_query}", 
generate a highly effective GitHub search query. Use only GitHub search syntax (e.g., language:python, keywords, filters). 
Return ONLY the optimized query with no additional explanation."""
    messages = [
        ("system", "You are a helpful research assistant specializing in GitHub search."),
        ("human", prompt)
    ]
    result = llm_groq.invoke(messages)
    logger.info(f"Enhanced Query: {result}")
    # Extract the query text (assuming it is returned in the 'content' field)
    return result.content if hasattr(result, "content") else str(result)


@tool
def fetch_github_repositories_tool(query: str, max_results: int = 1000, per_page: int = 100) -> List[Dict[str, Any]]:
    """
    Searches GitHub repositories using the given query.
    """
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
        response = requests.get(url, headers=gh_headers, params=params)
        if response.status_code != 200:
            logger.error(f"Error {response.status_code}: {response.json().get('message')}")
            break
        items = response.json().get('items', [])
        if not items:
            break
        for repo in items:
            full_name = repo.get("full_name", "")
            doc_content = fetch_repo_documentation(full_name)
            repositories.append({
                "title": repo.get("name", "No title available"),
                "link": repo.get("html_url", ""),
                "combined_doc": doc_content,
                "stars": repo.get("stargazers_count", 0),
                "full_name": full_name,
                "open_issues_count": repo.get("open_issues_count", 0)
            })
    logger.info(f"Fetched {len(repositories)} repositories from GitHub.")
    return repositories

@tool
def semantic_ranking_tool(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ranks candidates using SentenceTransformer and FAISS.
    """
    if not candidates:
        logger.info("No candidates provided for semantic ranking. Returning empty list.")
        return []
    docs = [repo.get("combined_doc", "") for repo in candidates]
    sem_model = SentenceTransformer("all-mpnet-base-v2")
    logger.info(f"Encoding {len(docs)} documents for dense retrieval...")
    doc_embeddings = sem_model.encode(docs, convert_to_numpy=True, show_progress_bar=True, batch_size=16)
    if doc_embeddings.ndim == 1:
        doc_embeddings = doc_embeddings.reshape(1, -1)
    else:
        norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        doc_embeddings = doc_embeddings / (norms + 1e-10)
    query_embedding = sem_model.encode(query, convert_to_numpy=True).reshape(1, -1)
    query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    query_embedding = query_embedding / (query_norm + 1e-10)
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings)
    k = min(100, len(candidates))
    distances, indices = index.search(query_embedding, k)
    for idx, score in zip(indices[0], distances[0]):
        candidates[idx]["semantic_similarity"] = score
    ranked = sorted(candidates, key=lambda x: x.get("semantic_similarity", 0), reverse=True)
    logger.info(f"Semantic ranking complete: {len(ranked)} candidates.")
    return ranked

@tool
def cross_encoder_rerank_tool(query: str, candidates: List[Dict[str, Any]], top_n: int = 50) -> List[Dict[str, Any]]:
    """
    Re-ranks candidates using a CrossEncoder.
    """
    if not candidates:
        logger.info("No candidates for cross-encoder reranking. Returning empty list.")
        return []
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, candidate["combined_doc"]] for candidate in candidates]
    scores = cross_encoder.predict(pairs, show_progress_bar=True)
    for candidate, score in zip(candidates, scores):
        candidate["cross_encoder_score"] = score
    reranked = sorted(candidates, key=lambda x: x["cross_encoder_score"], reverse=True)[:top_n]
    logger.info(f"Cross-encoder reranking complete: {len(reranked)} candidates.")
    return reranked

@tool
def filter_low_star_repos_tool(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filters out repositories with low star counts (unless they have high cross-encoder scores).
    """
    if not candidates:
        logger.info("No candidates for filtering. Returning empty list.")
        return []
    filtered = [repo for repo in candidates if repo["stars"] >= 50 or repo.get("cross_encoder_score", 0) >= 5.5]
    if not filtered:
        filtered = candidates
    logger.info(f"Filtered {len(filtered)} candidates after low-star filtering.")
    return filtered

@tool
def analyze_activity_tool(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Analyzes repository activity based on PRs, commits, and issues.
    """
    if not candidates:
        logger.info("No candidates for activity analysis. Returning empty list.")
        return []
    for repo in candidates:
        full_name = repo.get("full_name", "")
        pr_url = f"https://api.github.com/repos/{full_name}/pulls"
        pr_response = requests.get(pr_url, headers=gh_headers, params={"state": "open", "per_page": 100})
        pr_count = len(pr_response.json()) if pr_response.status_code == 200 else 0
        commits_url = f"https://api.github.com/repos/{full_name}/commits"
        commits_response = requests.get(commits_url, headers=gh_headers, params={"per_page": 1})
        if commits_response.status_code == 200 and commits_response.json():
            commit_date_str = commits_response.json()[0]["commit"]["committer"]["date"]
            commit_date = datetime.datetime.fromisoformat(commit_date_str.rstrip("Z"))
            days_diff = (datetime.datetime.utcnow() - commit_date).days
        else:
            days_diff = 999
        open_issues = repo.get("open_issues_count", 0)
        non_pr_issues = max(0, open_issues - pr_count)
        activity_score = (3 * pr_count) + non_pr_issues - (days_diff / 30)
        repo.update({"pr_count": pr_count, "latest_commit_days": days_diff, "activity_score": activity_score})
    logger.info("Activity analysis complete.")
    return candidates

@tool
def final_scoring_tool(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Combines semantic, cross-encoder, activity, and star scores for final ranking.
    """
    if not candidates:
        logger.info("No candidates for final scoring. Returning empty list.")
        return []
    semantic_scores = [repo.get("semantic_similarity", 0) for repo in candidates]
    cross_encoder_scores = [repo.get("cross_encoder_score", 0) for repo in candidates]
    activity_scores = [repo.get("activity_score", -100) for repo in candidates]
    star_scores = [math.log(repo.get("stars", 0) + 1) for repo in candidates]
    min_sem, max_sem = min(semantic_scores), max(semantic_scores)
    min_ce, max_ce = min(cross_encoder_scores), max(cross_encoder_scores)
    min_act, max_act = min(activity_scores), max(activity_scores)
    min_star, max_star = min(star_scores), max(star_scores)
    def normalize(val, min_val, max_val):
        if max_val - min_val == 0:
            return 0.5
        return (val - min_val) / (max_val - min_val)
    for repo in candidates:
        norm_sem = normalize(repo.get("semantic_similarity", 0), min_sem, max_sem)
        norm_ce = normalize(repo.get("cross_encoder_score", 0), min_ce, max_ce)
        norm_act = normalize(repo.get("activity_score", -100), min_act, max_act)
        norm_star = normalize(math.log(repo.get("stars", 0) + 1), min_star, max_star)
        repo["final_score"] = 0.3 * norm_ce + 0.2 * norm_sem + 0.2 * norm_act + 0.3 * norm_star
    ranked = sorted(candidates, key=lambda x: x["final_score"], reverse=True)
    logger.info(f"Final scoring complete: {len(ranked)} candidates.")
    return ranked

@tool
def justify_candidates_tool(candidates: List[Dict[str, Any]], top_n: int = 10) -> Dict[str, str]:
    """
    Generates a brief justification for each of the top candidates.
    """
    if not candidates:
        logger.info("No candidates for justification. Returning empty dictionary.")
        return {}
    justifications = {}
    for repo in candidates[:top_n]:
        prompt = f"""You are a highly knowledgeable AI research assistant. In one to two lines, explain why the repository titled "{repo['title']}" is a good match for a query on Chain of Thought prompting in large language models within a Python environment. Mention key factors such as documentation quality, activity, and community validation if relevant.

Repository Details:
- Stars: {repo['stars']}
- Semantic Similarity: {repo.get('semantic_similarity', 0):.4f}
- Cross-Encoder Score: {repo.get('cross_encoder_score', 0):.4f}
- Activity Score: {repo.get('activity_score', 0):.2f}

Provide a concise justification:"""
        messages = [
            ("system", "You are a highly knowledgeable AI research assistant that can succinctly justify repository matches."),
            ("human", prompt)
        ]
        result = llm_groq.invoke(messages)
        justifications[repo["title"]] = result
        logger.info(f"Justification for {repo['title']}: {result}")
    return justifications

# ------------------------------------------------------------------
# Workflow Definition using LangGraph
# ------------------------------------------------------------------
workflow = StateGraph(AgentState)

# Use the helper function to ensure we get a plain string for the enhanced query.
workflow.add_node("enhance_query", lambda state: {
    "enhanced_query": get_enhanced_query(state["original_query"])
})
workflow.add_node("set_github_query", lambda state: {
    "github_query": state["enhanced_query"] + " language:python"
})
workflow.add_node("fetch_repos", lambda state: {
    "candidates": fetch_github_repositories_tool.invoke({"query": state["github_query"]})
})
workflow.add_node("semantic_rank", lambda state: {
    "candidates": semantic_ranking_tool.invoke({
        "query": state["original_query"],
        "candidates": state["candidates"]
    })
})
workflow.add_node("cross_encoder_rerank", lambda state: {
    "candidates": cross_encoder_rerank_tool.invoke({
        "query": state["original_query"],
        "candidates": state["candidates"]
    })
})
workflow.add_node("filter_low_star", lambda state: {
    "candidates": filter_low_star_repos_tool.invoke({"candidates": state["candidates"]})
})
workflow.add_node("analyze_activity", lambda state: {
    "candidates": analyze_activity_tool.invoke({"candidates": state["candidates"]})
})
workflow.add_node("final_scoring", lambda state: {
    "final_ranked": final_scoring_tool.invoke({"candidates": state["candidates"]})
})
workflow.add_node("justify", lambda state: {
    "justifications": justify_candidates_tool.invoke({"candidates": state["final_ranked"]})
})

workflow.set_entry_point("enhance_query")
workflow.add_edge("enhance_query", "set_github_query")
workflow.add_edge("set_github_query", "fetch_repos")
workflow.add_edge("fetch_repos", "semantic_rank")
workflow.add_edge("semantic_rank", "cross_encoder_rerank")
workflow.add_edge("cross_encoder_rerank", "filter_low_star")
workflow.add_edge("filter_low_star", "analyze_activity")
workflow.add_edge("analyze_activity", "final_scoring")
workflow.add_edge("final_scoring", "justify")
workflow.add_edge("justify", END)

agent = workflow.compile()

# ------------------------------------------------------------------
# Execute the Agent Workflow
# ------------------------------------------------------------------
initial_state = {
    "original_query": "I am looking for finetuning gemini models."
}
result = agent.invoke(initial_state)

# ------------------------------------------------------------------
# Final Output
# ------------------------------------------------------------------
print("\n=== Final Ranked Repositories ===")
for rank, repo in enumerate(result["final_ranked"][:10], 1):
    print(f"Final Rank: {rank}")
    print(f"Title: {repo['title']}")
    print(f"Link: {repo['link']}")
    print(f"Stars: {repo['stars']}")
    print(f"Semantic Similarity: {repo.get('semantic_similarity', 0):.4f}")
    print(f"Cross-Encoder Score: {repo.get('cross_encoder_score', 0):.4f}")
    print(f"Activity Score: {repo.get('activity_score', 0):.2f}")
    print(f"Final Score: {repo.get('final_score', 0):.4f}")
    print(f"Justification: {result['justifications'].get(repo['title'], 'No justification available')}")
    print(f"Combined Doc Snippet: {repo['combined_doc'][:200]}...")
    print('-' * 80)
print("\n=== End of Results ===")
