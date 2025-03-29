# tools/ranking.py
import math
import logging

logger = logging.getLogger(__name__)



# tools/normalize.py
def normalize_scores(values):
    """
    Perform min–max normalization on a list of numeric values.
    Returns a list of values scaled to [0, 1]. If the range is zero,
    returns 0.5 for each value.
    """
    min_val = min(values)
    max_val = max(values)
    if max_val - min_val == 0:
        return [0.5 for _ in values]
    return [(val - min_val) / (max_val - min_val) for val in values]


def multi_factor_ranking(state, config):
    # Gather raw scores from filtered candidates.
    semantic_scores = [repo.get("semantic_similarity", 0) for repo in state.filtered_candidates]
    cross_encoder_scores = [repo.get("cross_encoder_score", 0) for repo in state.filtered_candidates]
    activity_scores = [repo.get("activity_score", -100) for repo in state.filtered_candidates]
    quality_scores = [repo.get("code_quality_score", 0) for repo in state.filtered_candidates]
    star_scores = [math.log(repo.get("stars", 0) + 1) for repo in state.filtered_candidates]
    
    # Normalize each set of scores using the helper function.
    norm_sem_scores = normalize_scores(semantic_scores)
    norm_ce_scores = normalize_scores(cross_encoder_scores)
    norm_act_scores = normalize_scores(activity_scores)
    norm_quality_scores = normalize_scores(quality_scores)
    norm_star_scores = normalize_scores(star_scores)
    
    # Define weights for each signal.
    weights = {
        "cross_encoder": 0.30,
        "semantic": 0.20,
        "activity": 0.15,
        "quality": 0.15,
        "stars": 0.20
    }
    
    # Combine the normalized scores using the defined weights.
    for idx, repo in enumerate(state.filtered_candidates):
        repo["final_score"] = (
            weights["cross_encoder"] * norm_ce_scores[idx] +
            weights["semantic"] * norm_sem_scores[idx] +
            weights["activity"] * norm_act_scores[idx] +
            weights["quality"] * norm_quality_scores[idx] +
            weights["stars"] * norm_star_scores[idx]
        )
    
    # Sort repositories in descending order of final_score.
    state.final_ranked = sorted(state.filtered_candidates, key=lambda x: x["final_score"], reverse=True)
    logger.info(f"Final multi-factor ranking computed for {len(state.final_ranked)} candidates.")
    return {"final_ranked": state.final_ranked}

