<h1 align="center">
  <img src="https://img.icons8.com/?size=100&id=118557&format=png&color=000000" width="72" style="vertical-align: middle;"/> DeepGit Modules
</h1>

This document provides detailed descriptions of each module in the DeepGit workflow.

### 1. Query Conversion (`tools/convert_query.py`)
- **Purpose:** Convert the raw user query into colon-separated search tags.
- **Mechanism:** Uses an LLM prompt to generate precise tags.
- **Outcome:** Updates `state.searchable_query`.

### 2. Repository Ingestion (`tools/github.py`)
- **Purpose:** Retrieve GitHub repositories based on search tags.
- **Mechanism:**  
  - Uses asynchronous HTTP calls (via `httpx.AsyncClient`) to query GitHub.
  - Fetches README files and additional markdown documentation.
  - Combines all documentation into `combined_doc`.
- **Outcome:** Populates `state.repositories` with repository metadata and documentation.

### 3. Neural Dense Retrieval (`tools/dense_retrieval.py`)
- **Purpose:** Compute semantic similarity between the user query and repository documentation.
- **Mechanism:**  
  - Encodes text using a SentenceTransformer.
  - Normalizes embeddings and uses FAISS to search for nearest neighbors.
- **Outcome:** Produces a sorted list (`state.semantic_ranked`) with similarity scores.

### 4. Cross-Encoder Re-Ranking (`tools/cross_encoder_reranking.py`)
- **Purpose:** Refine ranking by comparing the complete markdown documentation against the query.
- **Mechanism:**  
  - For short documentation, scores the full text directly.
  - For long documentation, splits it into chunks (with configurable chunk size and max length) and scores each chunk.
  - Uses the maximum score as the repository's final cross-encoder score.
- **Outcome:** Updates `state.reranked_candidates` with enhanced relevance scores.

### 5. Threshold Filtering (`tools/filtering.py`)
- **Purpose:** Filter out repositories that don't meet quality thresholds.
- **Mechanism:**  
  - Evaluates candidates based on star count and cross-encoder score.
  - Discards repositories failing to meet the thresholds.
- **Outcome:** Sets `state.filtered_candidates`.

### 6. Decision Maker (`tools/decision_maker.py`)
- **Purpose:** Decide if code quality analysis is needed.
- **Mechanism:**  
  - Uses an LLM prompt that evaluates the user's query and repository count.
  - Outputs a decision (1 to run analysis, 0 to skip).
- **Outcome:** Sets `state.run_code_analysis`.

### 7. Repository Activity Analysis (`tools/activity_analysis.py`)
- **Purpose:** Assess the repository's activity level.
- **Mechanism:**  
  - Fetches pull requests, commit dates, and open issues.
  - Computes an `activity_score` based on these metrics.
- **Outcome:** Populates `state.activity_candidates`.

### 8. Code Quality Analysis (`tools/code_quality.py`)
- **Purpose:** Evaluate code quality if required.
- **Mechanism:**  
  - Clones repositories locally.
  - Runs flake8 to count style errors.
  - Computes a score based on issues per file.
- **Outcome:** Populates `state.quality_candidates`.

### 9. Merge Analysis (`tools/merge_analysis.py`)
- **Purpose:** Combine results from the activity and code quality analyses.
- **Mechanism:**  
  - Merges candidates based on repository `full_name`.
- **Outcome:** Updates `state.filtered_candidates` with merged information.

### 10. Multi-Factor Ranking (`tools/ranking.py`)
- **Purpose:** Compute a final ranking score by combining various metrics.
- **Mechanism:**  
  - Normalizes scores (semantic, cross-encoder, activity, code quality, stars).
  - Applies predetermined weights and sums them to produce a final score.
- **Outcome:** Produces a sorted `state.final_ranked` list.

### 11. Output Presentation (`tools/output_presentation.py`)
- **Purpose:** Format and display the final ranked repositories.
- **Mechanism:**  
  - Constructs a string output with details of the top-ranked repositories.
- **Outcome:** Returns the final results in `state.final_results`.

---

