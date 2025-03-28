# tools/cross_encoder_reranking.py
import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

def cross_encoder_reranking(state, config):
    from agent import AgentConfiguration
    agent_config = AgentConfiguration.from_runnable_config(config)
    cross_encoder = CrossEncoder(agent_config.cross_encoder_model_name)
    candidates_for_rerank = state.semantic_ranked[:100]
    logger.info(f"Re-ranking {len(candidates_for_rerank)} candidates with cross-encoder...")
    
    def cross_encoder_rerank_func(query, candidates, top_n):
        pairs = [[query, candidate["combined_doc"]] for candidate in candidates]
        scores = cross_encoder.predict(pairs, show_progress_bar=True)
        for candidate, score in zip(candidates, scores):
            candidate["cross_encoder_score"] = score
        return sorted(candidates, key=lambda x: x["cross_encoder_score"], reverse=True)[:top_n]
    
    state.reranked_candidates = cross_encoder_rerank_func(
        state.user_query,
        candidates_for_rerank,
        int(agent_config.cross_encoder_top_n)
    )
    logger.info(f"Cross-encoder re-ranking complete: {len(state.reranked_candidates)} candidates remain.")
    return {"reranked_candidates": state.reranked_candidates}
