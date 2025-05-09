# tools/cross_encoder_reranking.py
import numpy as np
import logging
from typing import List, Dict, Any
from openai import OpenAI
from sentence_transformers import CrossEncoder
from tools.llm_config import PipelineType

logger = logging.getLogger(__name__)

def cross_encoder_reranking(state, config):
    """
    Re-ranks candidates using either OpenAI's chat completion API or SentenceTransformer
    for semantic similarity scoring.
    """
    from agent import AgentConfiguration
    agent_config = AgentConfiguration.from_runnable_config(config)
    
    # Get pipeline type from config
    pipeline_type = config.get("pipeline_type", PipelineType.OPENAI_PIPELINE)
    
    if pipeline_type == PipelineType.OPENAI_PIPELINE:
        return _openai_reranking(state, agent_config)
    else:
        return _sentence_transformer_reranking(state, agent_config)

def _openai_reranking(state, agent_config):
    """OpenAI-based reranking implementation"""
    client = OpenAI()
    
    # Use top candidates from semantic ranking
    candidates_for_rerank = state.semantic_ranked[:100]
    logger.info(f"Re-ranking {len(candidates_for_rerank)} candidates with OpenAI...")

    # Configuration for chunking
    CHUNK_SIZE = 2000        # characters per chunk
    MAX_DOC_LENGTH = 5000    # cap for long docs
    MIN_DOC_LENGTH = 200     # threshold for short docs

    def split_text(text, chunk_size=CHUNK_SIZE):
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    def get_similarity_score(query, doc):
        """Get similarity score using OpenAI's chat completion API."""
        try:
            response = client.chat.completions.create(
                model=agent_config.reranking_model,
                messages=[
                    {"role": "system", "content": "You are a semantic similarity scorer. Given a query and a document, score how relevant the document is to the query on a scale from 0 to 10. Return only the numerical score."},
                    {"role": "user", "content": f"Query: {query}\n\nDocument: {doc}\n\nScore (0-10):"}
                ],
                temperature=0.1,
                max_tokens=5
            )
            # Extract and parse the score
            score_text = response.choices[0].message.content.strip()
            try:
                score = float(score_text)
                return max(0, min(10, score))  # Clamp between 0 and 10
            except ValueError:
                logger.warning(f"Could not parse score: {score_text}")
                return 0.0
        except Exception as e:
            logger.error(f"Error getting similarity score: {e}")
            return 0.0

    def rerank_candidates(query, candidates, top_n):
        for candidate in candidates:
            doc = candidate.get("combined_doc", "")
            # Limit document length if needed
            if len(doc) > MAX_DOC_LENGTH:
                doc = doc[:MAX_DOC_LENGTH]
            
            try:
                if len(doc) < MIN_DOC_LENGTH:
                    # For very short docs, score directly
                    score = get_similarity_score(query, doc)
                    candidate["cross_encoder_score"] = score
                else:
                    # For longer docs, split into chunks and take max score
                    chunks = split_text(doc)
                    scores = [get_similarity_score(query, chunk) for chunk in chunks]
                    candidate["cross_encoder_score"] = max(scores) if scores else 0.0
            except Exception as e:
                logger.error(f"Error scoring candidate {candidate.get('full_name', 'unknown')}: {e}")
                candidate["cross_encoder_score"] = 0.0
        
        # Postprocessing: Normalize scores to 0-1 range
        all_scores = [candidate["cross_encoder_score"] for candidate in candidates]
        max_score = max(all_scores) if all_scores else 1.0
        if max_score > 0:
            for candidate in candidates:
                candidate["cross_encoder_score"] /= max_score

        # Return top N candidates sorted by cross_encoder_score (descending)
        return sorted(candidates, key=lambda x: x["cross_encoder_score"], reverse=True)[:top_n]

    state.reranked_candidates = rerank_candidates(
        state.user_query,
        candidates_for_rerank,
        int(agent_config.cross_encoder_top_n)
    )
    logger.info(f"OpenAI re-ranking complete: {len(state.reranked_candidates)} candidates remain.")
    return {"reranked_candidates": state.reranked_candidates}

def _sentence_transformer_reranking(state, agent_config):
    """SentenceTransformer-based reranking implementation"""
    # Load the cross-encoder model
    model = CrossEncoder(agent_config.sentence_transformer_model)
    
    # Use top candidates from semantic ranking
    candidates_for_rerank = state.semantic_ranked[:100]
    logger.info(f"Re-ranking {len(candidates_for_rerank)} candidates with SentenceTransformer...")

    # Prepare pairs for scoring
    pairs = [(state.user_query, doc.get("combined_doc", "")) for doc in candidates_for_rerank]
    
    # Get scores from the model
    scores = model.predict(pairs)
    
    # Attach scores to candidates
    for candidate, score in zip(candidates_for_rerank, scores):
        candidate["cross_encoder_score"] = float(score)
    
    # Sort by score and take top N
    state.reranked_candidates = sorted(
        candidates_for_rerank,
        key=lambda x: x["cross_encoder_score"],
        reverse=True
    )[:int(agent_config.cross_encoder_top_n)]
    
    logger.info(f"SentenceTransformer re-ranking complete: {len(state.reranked_candidates)} candidates remain.")
    return {"reranked_candidates": state.reranked_candidates}
