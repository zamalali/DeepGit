import logging
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def hybrid_dense_retrieval(state, config):
    """
    Performs advanced hybrid dense retrieval using ColBERTv2 embeddings (CPU-only)
    fused with BM25 sparse retrieval on the combined repository documentation.

    Args:
        state: Agent state containing `user_query` (str) and `repositories` (list of dicts with 'combined_doc').
        config: Runnable configuration dict, optionally containing `configurable` overrides.

    Returns:
        dict with key 'semantic_ranked' containing the repositories sorted by combined score.
    """
    # Extract parameters directly from the config dict without importing AgentConfiguration
    cfg = config.get("configurable", {}) if isinstance(config, dict) else {}
    colbert_model_name = cfg.get("colbert_model_name", "colbert-ir/colbertv2.0")
    alpha = cfg.get("retrieval_alpha", 0.7)

    logger.info(f"Loading ColBERT model '{colbert_model_name}' for advanced vector embeddings...")
    device = cfg.get("device", "cpu")
    tokenizer = AutoTokenizer.from_pretrained(colbert_model_name)
    colbert_model = AutoModel.from_pretrained(colbert_model_name)
    colbert_model.to(device)
    colbert_model.eval()

    # Gather documents
    docs = [repo.get("combined_doc", "") for repo in state.repositories]
    if not docs:
        logger.warning("No documentation found in any repository. Skipping dense retrieval.")
        state.semantic_ranked = []
        return {"semantic_ranked": state.semantic_ranked}

    def encode_colbert(text: str) -> np.ndarray:
        """
        Token-level normalized embeddings for a single text via ColBERT.
        Returns an array of shape (num_tokens, embedding_dim).
        """
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = colbert_model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(0)
        # Normalize each token embedding
        embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-10)
        return embeddings.cpu().numpy()

    # Encode the user query
    logger.info("Encoding user query using ColBERT model...")
    query_embeddings = encode_colbert(state.user_query)

    # Compute ColBERT-based scores for each document
    logger.info(f"Scoring {len(docs)} documents with ColBERT embeddings...")
    colbert_scores = []
    for idx, doc in enumerate(docs):
        if not doc.strip():
            colbert_scores.append(0.0)
            continue
        try:
            doc_embeddings = encode_colbert(doc)
            # similarity matrix: query tokens vs doc tokens
            sim_matrix = np.dot(query_embeddings, doc_embeddings.T)
            # for each query token, take its max match in the document
            max_per_query = sim_matrix.max(axis=1)
            score = float(max_per_query.sum())
            colbert_scores.append(score)
        except Exception as e:
            logger.error(f"Error in ColBERT scoring for doc {idx}: {e}")
            colbert_scores.append(0.0)

    colbert_arr = np.array(colbert_scores)
    c_min, c_max = colbert_arr.min(), colbert_arr.max()
    norm_colbert = (colbert_arr - c_min) / (c_max - c_min + 1e-10)

    # BM25 sparse retrieval
    tokenized_docs = [doc.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    query_tokens = state.user_query.split()
    bm25_scores = np.array(bm25.get_scores(query_tokens))
    b_min, b_max = bm25_scores.min(), bm25_scores.max()
    norm_bm25 = (bm25_scores - b_min) / (b_max - b_min + 1e-10)

    # Combine dense and sparse signals
    combined = alpha * norm_colbert + (1 - alpha) * norm_bm25

    # Attach scores and sort repositories
    for idx, repo in enumerate(state.repositories):
        repo["semantic_similarity"] = float(combined[idx])

    state.semantic_ranked = sorted(
        state.repositories,
        key=lambda x: x.get("semantic_similarity", 0),
        reverse=True
    )
    logger.info(f"Hybrid ColBERT retrieval complete: {len(state.semantic_ranked)} candidates ranked.")

    return {"semantic_ranked": state.semantic_ranked}
