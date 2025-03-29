# tools/activity_analysis.py
import os
import datetime
import logging
import requests

logger = logging.getLogger(__name__)

def get_commit_frequency(full_name, headers):
    """
    Returns the number of commits in the last 30 days.
    """
    since_date = (datetime.datetime.utcnow() - datetime.timedelta(days=30)).isoformat() + "Z"
    commits_url = f"https://api.github.com/repos/{full_name}/commits"
    commits_params = {"since": since_date, "per_page": 100}
    try:
        response = requests.get(commits_url, headers=headers, params=commits_params)
        if response.status_code == 200:
            commits = response.json()
            return len(commits)
    except Exception as e:
        logger.error(f"Error fetching commit frequency for {full_name}: {e}")
    return 0

def repository_activity_analysis(state, config):
    headers = {
        "Authorization": f"token {os.getenv('GITHUB_API_KEY')}",
        "Accept": "application/vnd.github.v3+json"
    }
    def analyze_repository_activity(repo):
        full_name = repo.get("full_name")
        # Pull Requests analysis
        pr_url = f"https://api.github.com/repos/{full_name}/pulls"
        pr_params = {"state": "open", "per_page": 100}
        pr_response = requests.get(pr_url, headers=headers, params=pr_params)
        pr_count = len(pr_response.json()) if pr_response.status_code == 200 else 0

        # Latest commit analysis
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

        # Issues analysis: subtract PRs from total open issues.
        open_issues = repo.get("open_issues_count", 0)
        non_pr_issues = max(0, open_issues - pr_count)

        # New: Commit frequency in the last 30 days.
        commit_frequency = get_commit_frequency(full_name, headers)
        
        # Combine signals into an activity score.
        # Here, we give weight to PR count, subtract a penalty for stale commits,
        # add non-PR issues, and add a bonus for higher commit frequency.
        activity_score = (3 * pr_count) + non_pr_issues - (days_diff / 30) + (0.1 * commit_frequency)
        # Optionally, store commit frequency for further ranking analysis.
        repo["commit_frequency"] = commit_frequency
        repo["pr_count"] = pr_count
        repo["latest_commit_days"] = days_diff
        repo["activity_score"] = activity_score
        return repo
    
    activity_list = []
    # It is assumed that activity analysis runs on filtered candidates.
    for repo in state.filtered_candidates:
        data = analyze_repository_activity(repo)
        activity_list.append(data)
    state.activity_candidates = activity_list
    logger.info("Repository activity analysis complete.")
    return {"activity_candidates": state.activity_candidates}
