# tools/dense_retrieval.py
import numpy as np
import faiss
import logging
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi  # Ensure this package is installed

logger = logging.getLogger(__name__)

def hybrid_dense_retrieval(state, config):
    from agent import AgentConfiguration
    agent_config = AgentConfiguration.from_runnable_config(config)
    sem_model = SentenceTransformer(agent_config.sem_model_name)
    
    docs = [repo.get("combined_doc", "") for repo in state.repositories]
    if not docs:
        logger.warning("No documents found. Skipping dense retrieval.")
        state.semantic_ranked = []
        return {"semantic_ranked": state.semantic_ranked}
    
    logger.info(f"Encoding {len(docs)} documents for dense retrieval...")
    # Compute dense embeddings and normalize
    doc_embeddings = sem_model.encode(docs, convert_to_numpy=True, show_progress_bar=True, batch_size=16)
    if doc_embeddings.ndim == 1:
        doc_embeddings = doc_embeddings.reshape(1, -1)
    norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    norm_doc_embeddings = doc_embeddings / (norms + 1e-10)
    
    # Compute query embedding and normalize
    query_embedding = sem_model.encode(state.user_query, convert_to_numpy=True)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    
    # Dense similarity scores (cosine similarity)
    dense_sim_scores = np.dot(norm_doc_embeddings, query_embedding.T).squeeze()  # shape (num_docs,)
    # Normalize dense scores
    dense_min, dense_max = dense_sim_scores.min(), dense_sim_scores.max()
    norm_dense_scores = (dense_sim_scores - dense_min) / (dense_max - dense_min + 1e-10)
    
    # BM25 sparse retrieval
    tokenized_docs = [doc.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    query_tokens = state.user_query.split()
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_scores = np.array(bm25_scores)
    bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
    norm_bm25_scores = (bm25_scores - bm25_min) / (bm25_max - bm25_min + 1e-10)
    
    # Combine dense and BM25 scores (weighted sum)
    alpha = 0.7  # weight for dense signal; (1-alpha) for BM25 signal
    combined_scores = alpha * norm_dense_scores + (1 - alpha) * norm_bm25_scores
    
    # Assign combined scores to repositories
    for idx, repo in enumerate(state.repositories):
        repo["semantic_similarity"] = float(combined_scores[idx])
    
    state.semantic_ranked = sorted(state.repositories, key=lambda x: x.get("semantic_similarity", 0), reverse=True)
    logger.info(f"Hybrid dense retrieval complete: {len(state.semantic_ranked)} candidates ranked.")
    return {"semantic_ranked": state.semantic_ranked}
