# tools/dense_retrieval.py
import numpy as np
import faiss
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

def neural_dense_retrieval(state, config):
    from agent import AgentConfiguration
    agent_config = AgentConfiguration.from_runnable_config(config)
    sem_model = SentenceTransformer(agent_config.sem_model_name)
    
    docs = [repo.get("combined_doc", "") for repo in state.repositories]
    if not docs:
        logger.warning("No documents found. Skipping dense retrieval.")
        state.semantic_ranked = []
        return {"semantic_ranked": state.semantic_ranked}
    logger.info(f"Encoding {len(docs)} documents for dense retrieval...")
    doc_embeddings = sem_model.encode(docs, convert_to_numpy=True, show_progress_bar=True, batch_size=16)
    if doc_embeddings.ndim == 1:
        doc_embeddings = doc_embeddings.reshape(1, -1)
    
    def normalize_embeddings(embeddings):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-10)
    
    doc_embeddings = normalize_embeddings(doc_embeddings)
    query_embedding = sem_model.encode(state.user_query, convert_to_numpy=True)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    query_embedding = normalize_embeddings(query_embedding)[0]
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings)
    k = min(int(agent_config.dense_retrieval_k), doc_embeddings.shape[0])
    D, I = index.search(np.expand_dims(query_embedding, axis=0), k)
    for idx, score in zip(I[0], D[0]):
        state.repositories[idx]["semantic_similarity"] = score
    state.semantic_ranked = sorted(state.repositories, key=lambda x: x.get("semantic_similarity", 0), reverse=True)
    logger.info(f"Dense retrieval complete: {len(state.semantic_ranked)} candidates ranked.")
    return {"semantic_ranked": state.semantic_ranked}
