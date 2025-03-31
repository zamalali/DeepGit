<h1 align="center">
  <img src="https://img.icons8.com/?size=100&id=118557&format=png&color=000000" width="72" style="vertical-align: middle;"/> DeepGit Architecture
</h1>

DeepGit leverages a state graph to process a user's query and deliver a ranked list of repositories. Each node in the graph handles a specific function. Below is an overview of the entire architecture:

1. **Query Conversion**  
   - **Function:** Converts the raw user query into colon-separated search tags using an LLM.
   - **Module:** `tools/convert_query.py`

2. **Repository Ingestion**  
   - **Function:** Uses the GitHub API (with asynchronous calls) to fetch repository metadata and documentation.
   - **Module:** `tools/github.py`  
   - **Details:**  
     - Fetches README and additional markdown files.
     - Combines the content into a single `combined_doc` for each repository.

3. **Neural Dense Retrieval**  
   - **Function:** Encodes repository documentation using a Sentence Transformer and computes semantic similarity with the query using FAISS.
   - **Module:** `tools/dense_retrieval.py`  
   - **Details:**  
     - Normalizes embeddings.
     - Returns a ranked list of candidates based on semantic similarity.

4. **Cross-Encoder Re-Ranking**  
   - **Function:** Reranks candidates by comparing the user query with the complete markdown documentation of each repository.
   - **Module:** `tools/cross_encoder_reranking.py`  
   - **Details:**  
     - For long documentation, splits text into chunks.
     - Aggregates chunk scores (using the maximum value) to produce a final score.

5. **Threshold Filtering**  
   - **Function:** Filters out repositories that do not meet certain thresholds (e.g., minimum stars, cross encoder score).
   - **Module:** `tools/filtering.py`

6. **Decision Maker**  
   - **Function:** Determines if code quality analysis should be run based on the query and repository count.
   - **Module:** `tools/decision_maker.py`

7. **Repository Activity Analysis**  
   - **Function:** Computes an activity score based on factors like pull requests, commits, and open issues.
   - **Module:** `tools/activity_analysis.py`

8. **Code Quality Analysis**  
   - **Function:** (Conditional) Clones repositories and uses static analysis (flake8) to score code quality.
   - **Module:** `tools/code_quality.py`

9. **Merge Analysis**  
   - **Function:** Merges results from activity and code quality analyses.
   - **Module:** `tools/merge_analysis.py`

10. **Multi-Factor Ranking**  
    - **Function:** Normalizes and weights multiple metrics (semantic similarity, cross encoder score, activity score, code quality score, stars) to compute a final ranking.
    - **Module:** `tools/ranking.py`

11. **Output Presentation**  
    - **Function:** Formats the final ranked repositories into a human-readable output.
    - **Module:** `tools/output_presentation.py`

These nodes are connected in the state graph defined in `agent.py`, ensuring smooth data flow from the initial query to the final presentation of results.

---

