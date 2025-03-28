# tools/activity_analysis.py
import os
import datetime
import logging
import requests

logger = logging.getLogger(__name__)

def repository_activity_analysis(state, config):
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
