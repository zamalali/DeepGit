# tools/github.py
import os
import base64
import logging
import asyncio
from pathlib import Path
import httpx
from tools.mcp_adapter import mcp_adapter  # Import our MCP adapter

logger = logging.getLogger(__name__)

# In-memory cache to store file content for given URLs
FILE_CONTENT_CACHE = {}

async def fetch_readme_content(repo_full_name: str, headers: dict, client: httpx.AsyncClient) -> str:
    readme_url = f"https://api.github.com/repos/{repo_full_name}/readme"
    try:
        response = await mcp_adapter.fetch(readme_url, headers=headers, client=client)
        if response.status_code == 200:
            readme_data = response.json()
            content = readme_data.get('content', '')
            if content:
                return base64.b64decode(content).decode('utf-8')
    except Exception as e:
        logger.error(f"Error fetching README for {repo_full_name}: {e}")
    return ""

async def fetch_file_content(download_url: str, client: httpx.AsyncClient) -> str:
    if download_url in FILE_CONTENT_CACHE:
        return FILE_CONTENT_CACHE[download_url]
    try:
        response = await mcp_adapter.fetch(download_url, client=client)
        if response.status_code == 200:
            text = response.text
            FILE_CONTENT_CACHE[download_url] = text
            return text
    except Exception as e:
        logger.error(f"Error fetching file from {download_url}: {e}")
    return ""

async def fetch_directory_markdown(repo_full_name: str, path: str, headers: dict, client: httpx.AsyncClient) -> str:
    md_content = ""
    url = f"https://api.github.com/repos/{repo_full_name}/contents/{path}"
    try:
        response = await mcp_adapter.fetch(url, headers=headers, client=client)
        if response.status_code == 200:
            items = response.json()
            tasks = []
            for item in items:
                if item["type"] == "file" and item["name"].lower().endswith(".md"):
                    tasks.append(fetch_file_content(item["download_url"], client))
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for item, content in zip(items, results):
                    if item["type"] == "file" and item["name"].lower().endswith(".md") and not isinstance(content, Exception):
                        md_content += f"\n\n# {item['name']}\n" + content
    except Exception as e:
        logger.error(f"Error fetching directory markdown for {repo_full_name}/{path}: {e}")
    return md_content

async def fetch_repo_documentation(repo_full_name: str, headers: dict, client: httpx.AsyncClient) -> str:
    doc_text = ""
    readme_task = asyncio.create_task(fetch_readme_content(repo_full_name, headers, client))
    root_url = f"https://api.github.com/repos/{repo_full_name}/contents"
    try:
        response = await mcp_adapter.fetch(root_url, headers=headers, client=client)
        if response.status_code == 200:
            items = response.json()
            tasks = []
            for item in items:
                if item["type"] == "file" and item["name"].lower().endswith(".md"):
                    if item["name"].lower() != "readme.md":
                        tasks.append(asyncio.create_task(fetch_file_content(item["download_url"], client)))
                elif item["type"] == "dir" and item["name"].lower() in ["docs", "documentation"]:
                    tasks.append(asyncio.create_task(fetch_directory_markdown(repo_full_name, item["name"], headers, client)))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if not isinstance(res, Exception):
                    doc_text += "\n\n" + res
    except Exception as e:
        logger.error(f"Error fetching repository contents for {repo_full_name}: {e}")
    readme = await readme_task
    if readme:
        doc_text = "# README\n" + readme + doc_text
    return doc_text if doc_text.strip() else "No documentation available."

async def fetch_github_repositories(query: str, max_results: int, per_page: int, headers: dict) -> list:
    url = "https://api.github.com/search/repositories"
    repositories = []
    num_pages = max_results // per_page
    async with httpx.AsyncClient() as client:
        for page in range(1, num_pages + 1):
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": per_page,
                "page": page
            }
            try:
                response = await mcp_adapter.fetch(url, headers=headers, params=params, client=client)
                if response.status_code != 200:
                    logger.error(f"Error {response.status_code}: {response.json().get('message')}")
                    break
                items = response.json().get('items', [])
                if not items:
                    break
                tasks = []
                for repo in items:
                    full_name = repo.get('full_name', '')
                    tasks.append(asyncio.create_task(fetch_repo_documentation(full_name, headers, client)))
                docs = await asyncio.gather(*tasks, return_exceptions=True)
                for repo, doc in zip(items, docs):
                    repo_link = repo['html_url']
                    full_name = repo.get('full_name', '')
                    clone_url = repo.get('clone_url', f"https://github.com/{full_name}.git")
                    star_count = repo.get('stargazers_count', 0)
                    repositories.append({
                        "title": repo.get('name', 'No title available'),
                        "link": repo_link,
                        "clone_url": clone_url,
                        "combined_doc": doc if not isinstance(doc, Exception) else "",
                        "stars": star_count,
                        "full_name": full_name,
                        "open_issues_count": repo.get('open_issues_count', 0)
                    })
            except Exception as e:
                logger.error(f"Error fetching repositories for query {query}: {e}")
                break
    logger.info(f"Fetched {len(repositories)} repositories for query '{query}'.")
    return repositories

async def ingest_github_repos_async(state, config) -> dict:
    headers = {
        "Authorization": f"token {os.getenv('GITHUB_API_KEY')}",
        "Accept": "application/vnd.github.v3+json"
    }
    keyword_list = [kw.strip() for kw in state.searchable_query.split(":") if kw.strip()]
    logger.info(f"Searchable keywords (raw): {keyword_list}")
    
    target_language = "python"
    filtered_keywords = []
    for kw in keyword_list:
        if kw.startswith("target-"):
            target_language = kw.split("target-")[-1]
        else:
            filtered_keywords.append(kw)
    keyword_list = filtered_keywords
    logger.info(f"Filtered keywords: {keyword_list} | Target language: {target_language}")
    
    all_repos = []
    from agent import AgentConfiguration
    agent_config = AgentConfiguration.from_runnable_config(config)
    tasks = []
    for keyword in keyword_list:
        query = f"{keyword} language:{target_language}"
        tasks.append(asyncio.create_task(fetch_github_repositories(query, agent_config.max_results, agent_config.per_page, headers)))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if not isinstance(result, Exception):
            all_repos.extend(result)
        else:
            logger.error(f"Error in fetching repositories for a keyword: {result}")
    seen = set()
    unique_repos = []
    for repo in all_repos:
        if repo["full_name"] not in seen:
            seen.add(repo["full_name"])
            unique_repos.append(repo)
    state.repositories = unique_repos
    logger.info(f"Total unique repositories fetched: {len(state.repositories)}")
    return {"repositories": state.repositories}

def ingest_github_repos(state, config):
    return asyncio.run(ingest_github_repos_async(state, config))
