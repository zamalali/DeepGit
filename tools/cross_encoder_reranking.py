import logging
import numpy as np
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

def cross_encoder_reranking(state, config):
    from agent import AgentConfiguration
    agent_config = AgentConfiguration.from_runnable_config(config)
    cross_encoder = CrossEncoder(agent_config.cross_encoder_model_name)
    candidates_for_rerank = state.semantic_ranked[:100]
    logger.info(f"Re-ranking {len(candidates_for_rerank)} candidates with cross-encoder...")

    # Configuration for chunking: adjust these values as needed.
    CHUNK_SIZE = 2000        # number of characters per chunk
    MAX_DOC_LENGTH = 5000      # process at most the first 5000 characters of the doc
    MIN_DOC_LENGTH = 200       # if doc is shorter than this, use as-is

    def split_text(text, chunk_size=CHUNK_SIZE):
        """Splits text into chunks of size 'chunk_size'."""
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    def cross_encoder_rerank_func(query, candidates, top_n):
        for candidate in candidates:
            doc = candidate.get("combined_doc", "")
            # If the doc is extremely long, cap it to MAX_DOC_LENGTH.
            if len(doc) > MAX_DOC_LENGTH:
                doc = doc[:MAX_DOC_LENGTH]
            try:
                if len(doc) < MIN_DOC_LENGTH:
                    # If the documentation is very short, score it directly.
                    score = cross_encoder.predict([[query, doc]], show_progress_bar=False)
                    candidate["cross_encoder_score"] = float(score[0])
                else:
                    # For longer documentation, split it into chunks.
                    chunks = split_text(doc)
                    pairs = [[query, chunk] for chunk in chunks]
                    # Predict without showing progress (to reduce verbosity).
                    scores = cross_encoder.predict(pairs, show_progress_bar=False)
                    # Use the maximum score from the chunks.
                    candidate["cross_encoder_score"] = float(np.max(np.array(scores))) if scores is not None else 0.0
            except Exception as e:
                logger.error(f"Error scoring candidate {candidate.get('full_name', 'unknown')}: {e}")
                candidate["cross_encoder_score"] = 0.0
        # Return candidates sorted by descending cross_encoder_score.
        return sorted(candidates, key=lambda x: x["cross_encoder_score"], reverse=True)[:top_n]

    state.reranked_candidates = cross_encoder_rerank_func(
        state.user_query,
        candidates_for_rerank,
        int(agent_config.cross_encoder_top_n)
    )
    logger.info(f"Cross-encoder re-ranking complete: {len(state.reranked_candidates)} candidates remain.")
    return {"reranked_candidates": state.reranked_candidates}
