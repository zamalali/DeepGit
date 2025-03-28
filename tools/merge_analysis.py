# tools/merge_analysis.py
import logging

logger = logging.getLogger(__name__)

def merge_analysis(state, config):
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
