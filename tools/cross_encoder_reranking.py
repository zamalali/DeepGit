# tools/cross_encoder_reranking.py
import numpy as np
import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

def cross_encoder_reranking(state, config):
    from agent import AgentConfiguration
    agent_config = AgentConfiguration.from_runnable_config(config)
    cross_encoder = CrossEncoder(agent_config.cross_encoder_model_name)
    # Use top candidates from semantic ranking (e.g., top 100)
    candidates_for_rerank = state.semantic_ranked[:100]
    logger.info(f"Re-ranking {len(candidates_for_rerank)} candidates with cross-encoder...")

    # Configuration for chunking
    CHUNK_SIZE = 2000        # characters per chunk
    MAX_DOC_LENGTH = 5000      # cap for long docs
    MIN_DOC_LENGTH = 200       # threshold for short docs

    def split_text(text, chunk_size=CHUNK_SIZE):
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    def cross_encoder_rerank_func(query, candidates, top_n):
        for candidate in candidates:
            doc = candidate.get("combined_doc", "")
            # Limit document length if needed.
            if len(doc) > MAX_DOC_LENGTH:
                doc = doc[:MAX_DOC_LENGTH]
            try:
                if len(doc) < MIN_DOC_LENGTH:
                    # For very short docs, score directly.
                    score = cross_encoder.predict([[query, doc]], show_progress_bar=False)
                    candidate["cross_encoder_score"] = float(score[0])
                else:
                    # For longer docs, split into chunks.
                    chunks = split_text(doc)
                    pairs = [[query, chunk] for chunk in chunks]
                    scores = cross_encoder.predict(pairs, show_progress_bar=False)
                    # Combine scores: weighted average of max and mean scores.
                    max_score = np.max(scores) if scores is not None else 0.0
                    avg_score = np.mean(scores) if scores is not None else 0.0
                    candidate["cross_encoder_score"] = float(0.5 * max_score + 0.5 * avg_score)
            except Exception as e:
                logger.error(f"Error scoring candidate {candidate.get('full_name', 'unknown')}: {e}")
                candidate["cross_encoder_score"] = 0.0
        
        # Postprocessing: Shift all scores upward if any are negative.
        all_scores = [candidate["cross_encoder_score"] for candidate in candidates]
        min_score = min(all_scores)
        if min_score < 0:
            shift = -min_score
            for candidate in candidates:
                candidate["cross_encoder_score"] += shift

        # Return top N candidates sorted by cross_encoder_score (descending)
        return sorted(candidates, key=lambda x: x["cross_encoder_score"], reverse=True)[:top_n]

    state.reranked_candidates = cross_encoder_rerank_func(
        state.user_query,
        candidates_for_rerank,
        int(agent_config.cross_encoder_top_n)
    )
    logger.info(f"Cross-encoder re-ranking complete: {len(state.reranked_candidates)} candidates remain.")
    return {"reranked_candidates": state.reranked_candidates}
