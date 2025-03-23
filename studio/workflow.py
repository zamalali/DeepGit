import os
import base64
import requests
import numpy as np
import datetime
import math
import logging
import getpass
from pathlib import Path
from dotenv import load_dotenv

import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

# =============================================================================
# Logging & Environment Setup
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables from a .env file located one directory above
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path)
if "GITHUB_API_KEY" not in os.environ:
    os.environ["GITHUB_API_KEY"] = getpass.getpass("Enter your GitHub API key: ")

# =============================================================================
# Helper Functions for GitHub API
# =============================================================================
def fetch_readme_content(repo_full_name, headers):
    readme_url = f"https://api.github.com/repos/{repo_full_name}/readme"
    response = requests.get(readme_url, headers=headers)
    if response.status_code == 200:
        readme_data = response.json()
        return base64.b64decode(readme_data['content']).decode('utf-8')
    else:
        return ""

def fetch_file_content(download_url):
    try:
        response = requests.get(download_url)
        if response.status_code == 200:
            return response.text
    except Exception as e:
        logger.error(f"Error fetching file: {e}")
    return ""

def fetch_directory_markdown(repo_full_name, path, headers):
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

def fetch_repo_documentation(repo_full_name, headers):
    doc_text = ""
    # 1. Fetch README.
    readme = fetch_readme_content(repo_full_name, headers)
    if readme:
        doc_text += "# README\n" + readme
    # 2. List root directory files.
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
                doc_text += f"\n\n# {item['name']} folder\n" + fetch_directory_markdown(repo_full_name, item["name"], headers)
    return doc_text if doc_text.strip() else "No documentation available."

def fetch_github_repositories(query, max_results=1000, per_page=100):
    url = "https://api.github.com/search/repositories"
    headers = {
        "Authorization": f"token {os.getenv('GITHUB_API_KEY')}",
        "Accept": "application/vnd.github.v3+json"
    }
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
            doc_content = fetch_repo_documentation(full_name, headers)
            star_count = repo.get('stargazers_count', 0)
            repositories.append({
                "title": repo.get('name', 'No title available'),
                "link": repo_link,
                "combined_doc": doc_content,
                "stars": star_count,
                "full_name": full_name,
                "open_issues_count": repo.get('open_issues_count', 0)
            })
    logger.info(f"Fetched {len(repositories)} repositories from GitHub.")
    return repositories

def analyze_repository_activity(repo, headers):
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

def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-10)

# =============================================================================
# LangGraph Node Definitions (each node represents a stage in the pipeline)
# =============================================================================
# NOTE: We assume that langgraph is installed and provides a Graph and Node base class.
from langgraph import Graph, Node  # adjust this import if your langgraph package layout is different

class GitHubFetchNode(Node):
    def run(self, query: str):
        repos = fetch_github_repositories(query)
        return repos

class DenseRetrievalNode(Node):
    def __init__(self, name: str, sem_model: SentenceTransformer):
        super().__init__(name=name)
        self.sem_model = sem_model

    def run(self, repos, user_query_text: str):
        docs = [repo.get("combined_doc", "") for repo in repos]
        logger.info(f"Encoding {len(docs)} documents for dense retrieval...")
        doc_embeddings = self.sem_model.encode(docs, convert_to_numpy=True, show_progress_bar=True, batch_size=16)
        doc_embeddings = normalize_embeddings(doc_embeddings)
        query_embedding = self.sem_model.encode(user_query_text, convert_to_numpy=True)
        query_embedding = normalize_embeddings(np.expand_dims(query_embedding, axis=0))[0]
        dim = doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(doc_embeddings)
        k = min(100, doc_embeddings.shape[0])
        D, I = index.search(np.expand_dims(query_embedding, axis=0), k)
        for idx, score in zip(I[0], D[0]):
            repos[idx]["semantic_similarity"] = score
        ranked_by_semantic = sorted(repos, key=lambda x: x.get("semantic_similarity", 0), reverse=True)
        logger.info(f"Stage 1 complete: {len(ranked_by_semantic)} candidates ranked by semantic similarity.")
        return ranked_by_semantic

class CrossEncoderNode(Node):
    def __init__(self, name: str, cross_encoder: CrossEncoder):
        super().__init__(name=name)
        self.cross_encoder = cross_encoder

    def run(self, repos, user_query_text: str, top_n: int = 50):
        # Use top 100 candidates from the previous stage for re-ranking.
        candidates = repos[:100]
        logger.info(f"Stage 2: Re-ranking {len(candidates)} candidates with cross-encoder...")
        pairs = [[user_query_text, candidate["combined_doc"]] for candidate in candidates]
        scores = self.cross_encoder.predict(pairs, show_progress_bar=True)
        for candidate, score in zip(candidates, scores):
            candidate["cross_encoder_score"] = score
        reranked_candidates = sorted(candidates, key=lambda x: x["cross_encoder_score"], reverse=True)[:top_n]
        logger.info(f"Stage 2 complete: {len(reranked_candidates)} candidates remain after cross-encoder re-ranking.")
        return reranked_candidates

class FilterNode(Node):
    def run(self, repos):
        filtered_candidates = []
        for repo in repos:
            if repo["stars"] < 50 and repo.get("cross_encoder_score", 0) < 5.5:
                continue
            filtered_candidates.append(repo)
        if not filtered_candidates:
            filtered_candidates = repos  # fallback if filtering is too strict
        logger.info(f"Stage 2.5 complete: {len(filtered_candidates)} candidates remain after filtering low-star repositories.")
        return filtered_candidates

class ActivityAnalysisNode(Node):
    def __init__(self, name: str, gh_headers: dict):
        super().__init__(name=name)
        self.gh_headers = gh_headers

    def run(self, repos):
        for repo in repos:
            activity_data = analyze_repository_activity(repo, self.gh_headers)
            repo.update(activity_data)
        logger.info("Stage 3 complete: Activity analysis done for filtered candidates.")
        return repos

class ScoreCombinationNode(Node):
    def run(self, repos):
        semantic_scores = [repo.get("semantic_similarity", 0) for repo in repos]
        cross_encoder_scores = [repo.get("cross_encoder_score", 0) for repo in repos]
        activity_scores = [repo.get("activity_score", -100) for repo in repos]
        star_scores = [math.log(repo.get("stars", 0) + 1) for repo in repos]  # log transform

        min_sem, max_sem = min(semantic_scores), max(semantic_scores)
        min_ce, max_ce = min(cross_encoder_scores), max(cross_encoder_scores)
        min_act, max_act = min(activity_scores), max(activity_scores)
        min_star, max_star = min(star_scores), max(star_scores)

        def normalize(val, min_val, max_val):
            if max_val - min_val == 0:
                return 0.5
            return (val - min_val) / (max_val - min_val)

        for repo in repos:
            norm_sem = normalize(repo.get("semantic_similarity", 0), min_sem, max_sem)
            norm_ce = normalize(repo.get("cross_encoder_score", 0), min_ce, max_ce)
            norm_act = normalize(repo.get("activity_score", -100), min_act, max_act)
            norm_star = normalize(math.log(repo.get("stars", 0) + 1), min_star, max_star)
            # Weights: 30% cross-encoder, 20% semantic, 20% activity, 30% stars.
            repo["final_score"] = 0.3 * norm_ce + 0.2 * norm_sem + 0.2 * norm_act + 0.3 * norm_star

        final_ranked = sorted(repos, key=lambda x: x["final_score"], reverse=True)
        logger.info(f"Stage 4 complete: Final ranking computed for {len(final_ranked)} candidates.")
        return final_ranked

class OutputNode(Node):
    def run(self, repos):
        print("\n=== Final Ranked Repositories ===")
        for rank, repo in enumerate(repos[:10], 1):
            print(f"Final Rank: {rank}")
            print(f"Title: {repo['title']}")
            print(f"Link: {repo['link']}")
            print(f"Stars: {repo['stars']}")
            print(f"Semantic Similarity: {repo.get('semantic_similarity', 0):.4f}")
            print(f"Cross-Encoder Score: {repo.get('cross_encoder_score', 0):.4f}")
            print(f"Activity Score: {repo.get('activity_score', 0):.2f}")
            print(f"Final Score: {repo.get('final_score', 0):.4f}")
            print(f"Combined Doc Snippet: {repo['combined_doc'][:200]}...")
            print('-' * 80)
        print("\n=== End of Results ===")
        return "Output complete"

# =============================================================================
# Main Pipeline using LangGraph
# =============================================================================
def main():
    # Initialize models
    sem_model = SentenceTransformer("all-mpnet-base-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    gh_headers = {
        "Authorization": f"token {os.getenv('GITHUB_API_KEY')}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Define your user query and GitHub search query.
    user_query_text = (
        "I am researching the application of Chain of Thought prompting for improving reasoning in large language models within a Python environment."
    )
    github_query = "Chain of Thought prompting language:python"
    
    # Create a LangGraph and add nodes
    graph = Graph(name="Deep GitHub Research Agent")
    
    github_node = GitHubFetchNode(name="GitHub Fetch Node")
    dense_node = DenseRetrievalNode(name="Dense Retrieval Node", sem_model=sem_model)
    cross_encoder_node = CrossEncoderNode(name="Cross Encoder Node", cross_encoder=cross_encoder)
    filter_node = FilterNode(name="Filter Node")
    activity_node = ActivityAnalysisNode(name="Activity Analysis Node", gh_headers=gh_headers)
    score_node = ScoreCombinationNode(name="Score Combination Node")
    output_node = OutputNode(name="Output Node")
    
    # (Optional) Here you could connect nodes with graph.connect() if you wish to register data flow;
    # for this example, we are running the nodes sequentially.
    graph.add_node(github_node)
    graph.add_node(dense_node)
    graph.add_node(cross_encoder_node)
    graph.add_node(filter_node)
    graph.add_node(activity_node)
    graph.add_node(score_node)
    graph.add_node(output_node)
    
    # Execute the pipeline sequentially:
    repos = github_node.run(github_query)
    repos = dense_node.run(repos, user_query_text)
    repos = cross_encoder_node.run(repos, user_query_text)
    repos = filter_node.run(repos)
    repos = activity_node.run(repos)
    final_ranked = score_node.run(repos)
    output_node.run(final_ranked)
    
    # Finally, visualize the graph in LangGraph Studio.
    # This call should open or generate a visualization that you can then view in LangGraph Studio.
    graph.visualize()

if __name__ == "__main__":
    main()
