# tools/code_quality.py
import os
import subprocess
import tempfile
import shutil
import stat
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

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

def code_quality_analysis(state, config):
    if not getattr(state, "run_code_analysis", False):
        logger.info("Skipping code quality analysis as per decision maker.")
        state.quality_candidates = []
        return {"quality_candidates": state.quality_candidates}
    quality_list = []
    for repo in state.filtered_candidates:
        if "clone_url" not in repo:
            repo["clone_url"] = f"https://github.com/{repo['full_name']}.git"
        updated_repo = analyze_code_quality(repo)
        quality_list.append(updated_repo)
    state.quality_candidates = quality_list
    logger.info("Code quality analysis complete.")
    return {"quality_candidates": state.quality_candidates}
