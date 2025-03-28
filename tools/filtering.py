# tools/filtering.py
import logging

logger = logging.getLogger(__name__)

def threshold_filtering(state, config):
    from agent import AgentConfiguration
    agent_config = AgentConfiguration.from_runnable_config(config)
    filtered = []
    for repo in state.reranked_candidates:
        if repo["stars"] < agent_config.min_stars and repo.get("cross_encoder_score", 0) < agent_config.cross_encoder_threshold:
            continue
        filtered.append(repo)
    if not filtered:
        filtered = state.reranked_candidates
    state.filtered_candidates = filtered
    logger.info(f"Filtering complete: {len(state.filtered_candidates)} candidates remain.")
    return {"filtered_candidates": state.filtered_candidates}
