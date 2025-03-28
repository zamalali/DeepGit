# tools/ranking.py
import math
import logging

logger = logging.getLogger(__name__)

def multi_factor_ranking(state, config):
    semantic_scores = [repo.get("semantic_similarity", 0) for repo in state.filtered_candidates]
    cross_encoder_scores = [repo.get("cross_encoder_score", 0) for repo in state.filtered_candidates]
    activity_scores = [repo.get("activity_score", -100) for repo in state.filtered_candidates]
    quality_scores = [repo.get("code_quality_score", 0) for repo in state.filtered_candidates]
    star_scores = [math.log(repo.get("stars", 0) + 1) for repo in state.filtered_candidates]

    min_sem, max_sem = min(semantic_scores), max(semantic_scores)
    min_ce, max_ce = min(cross_encoder_scores), max(cross_encoder_scores)
    min_act, max_act = min(activity_scores), max(activity_scores)
    min_quality, max_quality = min(quality_scores), max(quality_scores)
    min_star, max_star = min(star_scores), max(star_scores)

    def normalize(val, min_val, max_val):
        if max_val - min_val == 0:
            return 0.5
        return (val - min_val) / (max_val - min_val)

    for repo in state.filtered_candidates:
        norm_sem = normalize(repo.get("semantic_similarity", 0), min_sem, max_sem)
        norm_ce = normalize(repo.get("cross_encoder_score", 0), min_ce, max_ce)
        norm_act = normalize(repo.get("activity_score", -100), min_act, max_act)
        norm_quality = normalize(repo.get("code_quality_score", 0), min_quality, max_quality)
        norm_star = normalize(math.log(repo.get("stars", 0) + 1), min_star, max_star)
        # Weights: Cross-encoder 0.30, Semantic 0.20, Activity 0.15, Code Quality 0.15, Stars 0.20.
        repo["final_score"] = (0.30 * norm_ce +
                               0.20 * norm_sem +
                               0.15 * norm_act +
                               0.15 * norm_quality +
                               0.20 * norm_star)
    state.final_ranked = sorted(state.filtered_candidates, key=lambda x: x["final_score"], reverse=True)
    logger.info(f"Final multi-factor ranking computed for {len(state.final_ranked)} candidates.")
    return {"final_ranked": state.final_ranked}
