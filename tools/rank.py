import os
import base64
import requests
import numpy as np
import datetime
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import getpass
import math
import logging
from dotenv import load_dotenv
from pathlib import Path
from langchain_groq import ChatGroq

# ---------------------------
# Environment and .env Setup
# ---------------------------
# Resolve the path to the root directory's .env file
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=str(dotenv_path))

if "GITHUB_API_KEY" not in os.environ:
    os.environ["GITHUB_API_KEY"] = getpass.getpass("Enter your GitHub API key: ")

# ---------------------------
# Logging Setup
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------
# ChatGroq Integration Setup (for query enhancement only)
# ---------------------------
llm_groq = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=512,
    timeout=30,
    max_retries=2
)

def enhance_query(original_query):
    prompt = f"""You are an expert research assistant. Given the query: "{original_query}", 
please enhance and expand it by adding relevant technical keywords, recent research context, 
and details specifically related to the application of Chain of Thought prompting in large language models within a Python environment.
Provide the refined query text."""
    messages = [
        ("system", "You are a helpful research assistant specializing in AI and software research."),
        ("human", prompt)
    ]
    result = llm_groq.invoke(messages)
    return result

def justify_candidate(candidate):
    prompt = f"""You are a highly knowledgeable AI research assistant. In one to two lines, explain why the repository titled "{candidate['title']}" is a good match for a query on Chain of Thought prompting in large language models within a Python environment. Mention key factors such as documentation quality, activity, and community validation if relevant.

Repository Details:
- Stars: {candidate['stars']}
- Semantic Similarity: {candidate.get('semantic_similarity', 0):.4f}
- Cross-Encoder Score: {candidate.get('cross_encoder_score', 0):.4f}
- Activity Score: {candidate.get('activity_score', 0):.2f}

Provide a concise justification:"""
    messages = [
        ("system", "You are a highly knowledgeable AI research assistant that can succinctly justify repository matches."),
        ("human", prompt)
    ]
    result = llm_groq.invoke(messages)
    return result

# ---------------------------
# GitHub API Helper Functions
# ---------------------------
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
            "q": query,  # e.g., "Chain of Thought prompting language:python"
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

# ---------------------------
# Stage 0: Query Enhancement using ChatGroq
# ---------------------------
logger.info("Enhancing query using ChatGroq...")
# Define the original query (must be defined before use)
user_query_text = """
I am researching the application of Chain of Thought prompting for improving reasoning in large language models within a Python environment.
"""
original_query = user_query_text.strip()
enhanced_query = enhance_query(original_query)
logger.info(f"Enhanced Query: {enhanced_query}")
# Append language filter to the enhanced query
github_query = enhanced_query + " language:python"
logger.info(f"Using GitHub query: {github_query}")

# ---------------------------
# Stage 1: Dense Retrieval with FAISS
# ---------------------------
sem_model = SentenceTransformer("all-mpnet-base-v2")
repos = fetch_github_repositories(github_query)
docs = [repo.get("combined_doc", "") for repo in repos]
logger.info(f"Encoding {len(docs)} documents for dense retrieval...")
doc_embeddings = sem_model.encode(docs, convert_to_numpy=True, show_progress_bar=True, batch_size=16)

def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-10)

doc_embeddings = normalize_embeddings(doc_embeddings)
query_embedding = sem_model.encode(user_query_text, convert_to_numpy=True)
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

# ---------------------------
# Stage 2: Re-Ranking with Cross-Encoder
# ---------------------------
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def cross_encoder_rerank(query, candidates, top_n=50):
    pairs = [[query, candidate["combined_doc"]] for candidate in candidates]
    scores = cross_encoder.predict(pairs, show_progress_bar=True)
    for candidate, score in zip(candidates, scores):
        candidate["cross_encoder_score"] = score
    return sorted(candidates, key=lambda x: x["cross_encoder_score"], reverse=True)[:top_n]

candidates_for_rerank = ranked_by_semantic[:100]
logger.info(f"Stage 2: Re-ranking {len(candidates_for_rerank)} candidates with cross-encoder...")
reranked_candidates = cross_encoder_rerank(user_query_text, candidates_for_rerank, top_n=50)
logger.info(f"Stage 2 complete: {len(reranked_candidates)} candidates remain after cross-encoder re-ranking.")

# ---------------------------
# Stage 2.5: Filtering Low-Star Repositories
# ---------------------------
filtered_candidates = []
for repo in reranked_candidates:
    if repo["stars"] < 50 and repo.get("cross_encoder_score", 0) < 5.5:
        continue
    filtered_candidates.append(repo)
if not filtered_candidates:
    filtered_candidates = reranked_candidates  # fallback if filtering is too strict
logger.info(f"Stage 2.5 complete: {len(filtered_candidates)} candidates remain after filtering low-star repositories.")

# ---------------------------
# Stage 3: Activity Analysis
# ---------------------------
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

gh_headers = {
    "Authorization": f"token {os.getenv('GITHUB_API_KEY')}",
    "Accept": "application/vnd.github.v3+json"
}
for repo in filtered_candidates:
    activity_data = analyze_repository_activity(repo, gh_headers)
    repo.update(activity_data)
logger.info("Stage 3 complete: Activity analysis done for filtered candidates.")

# ---------------------------
# Stage 4: Combine Scores for Final Ranking (Including Stars)
# ---------------------------
semantic_scores = [repo.get("semantic_similarity", 0) for repo in filtered_candidates]
cross_encoder_scores = [repo.get("cross_encoder_score", 0) for repo in filtered_candidates]
activity_scores = [repo.get("activity_score", -100) for repo in filtered_candidates]
star_scores = [math.log(repo.get("stars", 0) + 1) for repo in filtered_candidates]  # log transform

min_sem, max_sem = min(semantic_scores), max(semantic_scores)
min_ce, max_ce = min(cross_encoder_scores), max(cross_encoder_scores)
min_act, max_act = min(activity_scores), max(activity_scores)
min_star, max_star = min(star_scores), max(star_scores)

def normalize(val, min_val, max_val):
    if max_val - min_val == 0:
        return 0.5
    return (val - min_val) / (max_val - min_val)

for repo in filtered_candidates:
    norm_sem = normalize(repo.get("semantic_similarity", 0), min_sem, max_sem)
    norm_ce = normalize(repo.get("cross_encoder_score", 0), min_ce, max_ce)
    norm_act = normalize(repo.get("activity_score", -100), min_act, max_act)
    norm_star = normalize(math.log(repo.get("stars", 0) + 1), min_star, max_star)
    # Weights: 30% cross-encoder, 20% semantic, 20% activity, 30% stars.
    repo["final_score"] = 0.3 * norm_ce + 0.2 * norm_sem + 0.2 * norm_act + 0.3 * norm_star

final_ranked = sorted(filtered_candidates, key=lambda x: x["final_score"], reverse=True)
logger.info(f"Stage 4 complete: Final ranking computed for {len(final_ranked)} candidates.")

# ---------------------------
# Final Output
# ---------------------------
print("\n=== Final Ranked Repositories ===")
for rank, repo in enumerate(final_ranked[:10], 1):
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
