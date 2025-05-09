import logging
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModel
import torch
from tools.llm_config import PipelineType

logger = logging.getLogger(__name__)


def hybrid_dense_retrieval(state, config):
    """
    Performs hybrid dense retrieval using either OpenAI embeddings or ColBERT embeddings
    combined with BM25 sparse retrieval.
    """
    from agent import AgentConfiguration
    agent_config = AgentConfiguration.from_runnable_config(config)
    
    # Get pipeline type from config
    pipeline_type = config.get("pipeline_type", PipelineType.OPENAI_PIPELINE)
    
    # If bm25_ranked is not populated, use repositories
    if not state.bm25_ranked:
        logger.info("BM25 ranking not found, using repositories directly")
        state.bm25_ranked = state.repositories
        # Add dummy BM25 scores
        for doc in state.bm25_ranked:
            doc["bm25_score"] = 0.5  # Default score
    
    if pipeline_type == PipelineType.OPENAI_PIPELINE:
        return _openai_dense_retrieval(state, agent_config)
    else:
        return _colbert_dense_retrieval(state, agent_config)

def _openai_dense_retrieval(state, agent_config):
    """OpenAI-based dense retrieval implementation"""
    client = OpenAI()
    
    def get_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
        # Truncate texts to fit within token limits
        max_tokens = 8191  # Maximum tokens for text-embedding-3-small
        truncated_texts = [text[:max_tokens * 4] for text in texts]  # Rough estimate: 4 chars per token
        
        try:
            response = client.embeddings.create(
                model=model,
                input=truncated_texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return [[0.0] * 1536] * len(texts)  # Return zero vectors as fallback
    
    # Get embeddings for query and documents
    query_embedding = get_embeddings([state.user_query], agent_config.embedding_model)[0]
    doc_embeddings = get_embeddings([doc.get("combined_doc", "") for doc in state.bm25_ranked], agent_config.embedding_model)
    
    # Compute cosine similarity
    query_norm = np.linalg.norm(query_embedding)
    doc_norms = np.array([np.linalg.norm(doc_emb) for doc_emb in doc_embeddings])
    
    # Avoid division by zero
    if query_norm == 0 or np.any(doc_norms == 0):
        openai_scores = np.zeros(len(doc_embeddings))
    else:
        similarities = np.dot(doc_embeddings, query_embedding) / (doc_norms * query_norm)
        # Normalize to 0-1 range and ensure we don't have any NaN values
        openai_scores = np.clip((similarities + 1) / 2, 0, 1)
    
    # Log some statistics about the scores
    logger.info(f"OpenAI semantic scores - min: {np.min(openai_scores):.4f}, max: {np.max(openai_scores):.4f}, mean: {np.mean(openai_scores):.4f}")
    
    # Combine with BM25 scores
    alpha = 0.5  # Weight for dense vs sparse scores
    for i, doc in enumerate(state.bm25_ranked):
        doc["semantic_similarity"] = float(openai_scores[i])
        doc["hybrid_score"] = alpha * doc["semantic_similarity"] + (1 - alpha) * doc["bm25_score"]
    
    # Sort by hybrid score
    state.semantic_ranked = sorted(state.bm25_ranked, key=lambda x: x["hybrid_score"], reverse=True)
    logger.info(f"OpenAI hybrid retrieval complete: {len(state.semantic_ranked)} candidates ranked.")
    return {"semantic_ranked": state.semantic_ranked}

def _colbert_dense_retrieval(state, agent_config):
    """ColBERT-based dense retrieval implementation"""
    # Load ColBERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(agent_config.colbert_model)
    model = AutoModel.from_pretrained(agent_config.colbert_model)
    model.eval()
    
    def get_colbert_embeddings(texts: List[str]) -> torch.Tensor:
        # Tokenize and get embeddings
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    
    # Get embeddings for query and documents
    query_embedding = get_colbert_embeddings([state.user_query])
    doc_embeddings = get_colbert_embeddings([doc.get("combined_doc", "") for doc in state.bm25_ranked])
    
    # Compute cosine similarity
    query_norm = torch.norm(query_embedding, dim=1)
    doc_norms = torch.norm(doc_embeddings, dim=1)
    
    # Avoid division by zero
    if query_norm == 0 or torch.any(doc_norms == 0):
        colbert_scores = torch.zeros(len(doc_embeddings))
    else:
        similarities = torch.mm(doc_embeddings, query_embedding.t()).squeeze() / (doc_norms * query_norm)
        # Normalize to 0-1 range and ensure we don't have any NaN values
        colbert_scores = torch.clamp((similarities + 1) / 2, 0, 1)
    
    # Log some statistics about the scores
    logger.info(f"ColBERT semantic scores - min: {torch.min(colbert_scores):.4f}, max: {torch.max(colbert_scores):.4f}, mean: {torch.mean(colbert_scores):.4f}")
    
    # Combine with BM25 scores
    alpha = 0.5  # Weight for dense vs sparse scores
    for i, doc in enumerate(state.bm25_ranked):
        doc["semantic_similarity"] = float(colbert_scores[i])
        doc["hybrid_score"] = alpha * doc["semantic_similarity"] + (1 - alpha) * doc["bm25_score"]
    
    # Sort by hybrid score
    state.semantic_ranked = sorted(state.bm25_ranked, key=lambda x: x["hybrid_score"], reverse=True)
    logger.info(f"ColBERT hybrid retrieval complete: {len(state.semantic_ranked)} candidates ranked.")
    return {"semantic_ranked": state.semantic_ranked}
