import os
import base64
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from pathlib import Path

# Resolve the path to the root directory
dotenv_path = Path(__file__).resolve().parent.parent / ".env"

# Load the .env file
load_dotenv(dotenv_path)
# GitHub API Setup
def fetch_github_repositories(query, max_results=10):
    """
    Searches GitHub repositories and retrieves links and README content.

    Parameters:
    - query (str): Search query for GitHub repositories.
    - max_results (int): Maximum number of results to fetch.

    Returns:
    - List[dict]: List of repositories with link, title, and README content.
    """
    url = "https://api.github.com/search/repositories"
    headers = {
        "Authorization": f"token {os.getenv('GITHUB_API_KEY')}",
        "Accept": "application/vnd.github.v3+json"
    }
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": max_results
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.json().get('message')}")
        return []

    repo_list = []
    for repo in response.json().get('items', []):
        repo_link = repo['html_url']
        readme_content = fetch_readme_content(repo['full_name'], headers)
        repo_list.append({
            "title": repo.get('name', 'No title available'),
            "link": repo_link,
            "readme": readme_content
        })
    return repo_list

# Fetch README content for each repository
def fetch_readme_content(repo_full_name, headers):
    readme_url = f"https://api.github.com/repos/{repo_full_name}/readme"
    response = requests.get(readme_url, headers=headers)

    if response.status_code == 200:
        readme_data = response.json()
        return base64.b64decode(readme_data['content']).decode('utf-8')
    else:
        return "No README available"

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# User's Abstract
my_abstract = """
I am researching the application of Chain of Thought (CoT) prompting in improving the performance of large language models (LLMs) within a ROS Environment.
"""
# Embed the user's abstract
my_abstract_embedding = model.encode(my_abstract)

# GitHub query and fetch repositories
query = "Chain of Thought prompting"
repositories = fetch_github_repositories(query)

# Initialize a list for storing similarities
similarities = []

# Compute embeddings and cosine similarity for each README content
for repo in repositories:
    readme_text = repo.get("readme", "")
    if readme_text:  # Only embed if README text is available
        readme_embedding = model.encode(readme_text)
        similarity = np.dot(readme_embedding, my_abstract_embedding) / (np.linalg.norm(readme_embedding) * np.linalg.norm(my_abstract_embedding))
        similarities.append({
            "title": repo['title'],
            "link": repo['link'],
            "readme": readme_text,
            "similarity": similarity
        })

# Sort repositories by similarity (highest first)
ranked_repositories = sorted(similarities, key=lambda x: x['similarity'], reverse=True)

# Display the top-ranked repositories
for rank, repo in enumerate(ranked_repositories, 1):
    print(f"Rank: {rank}")
    print(f"Title: {repo['title']}")
    print(f"Link: {repo['link']}")
    print(f"Similarity: {repo['similarity']:.4f}")
    print(f"README: {repo['readme'][:200]}...")  # Display a snippet of README
    print('-' * 80)

