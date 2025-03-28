# tools/github.py
import os
import base64
import requests
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def fetch_readme_content(repo_full_name, headers):
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
    readme = fetch_readme_content(repo_full_name, headers)
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
                doc_text += f"\n\n# {item['name']} folder\n" + fetch_directory_markdown(repo_full_name, item["name"], headers)
    return doc_text if doc_text.strip() else "No documentation available."

def fetch_github_repositories(query, max_results, per_page, headers):
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
            doc_content = fetch_repo_documentation(full_name, headers)
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

def ingest_github_repos(state, config):
    headers = {
        "Authorization": f"token {os.getenv('GITHUB_API_KEY')}",
        "Accept": "application/vnd.github.v3+json"
    }
    keyword_list = [kw.strip() for kw in state.searchable_query.split(":") if kw.strip()]
    logger.info(f"Searchable keywords: {keyword_list}")
    all_repos = []
    # Import AgentConfiguration from agent.py.
    from agent import AgentConfiguration
    agent_config = AgentConfiguration.from_runnable_config(config)
    for keyword in keyword_list:
        query = f"{keyword} language:python"
        repos = fetch_github_repositories(query, agent_config.max_results, agent_config.per_page, headers)
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
