# tools/output_presentation.py
"""
def output_presentation(state, config):
    results_str = "\n=== Final Ranked Repositories ===\n"
    top_n = 10
    for rank, repo in enumerate(state.final_ranked[:top_n], 1):
        results_str += f"\nFinal Rank: {rank}\n"
        results_str += f"Title: {repo['title']}\n"
        results_str += f"Link: {repo['link']}\n"
        results_str += f"Stars: {repo['stars']}\n"
        results_str += f"Semantic Similarity: {repo.get('semantic_similarity', 0):.4f}\n"
        results_str += f"Cross-Encoder Score: {repo.get('cross_encoder_score', 0):.4f}\n"
        results_str += f"Activity Score: {repo.get('activity_score', 0):.2f}\n"
        results_str += f"Code Quality Score: {repo.get('code_quality_score', 0)}\n"
        results_str += f"Final Score: {repo.get('final_score', 0):.4f}\n"
        results_str += f"Combined Doc Snippet: {repo['combined_doc'][:200]}...\n"
        results_str += '-' * 80 + "\n"
    return {"final_results": results_str}
"""

def output_presentation(state, config):
    results_str = "\n=== Final Ranked Repositories ===\n"
    top_n = 10
    for rank, repo in enumerate(state.final_ranked[:top_n], 1):
        results_str += f"\nFinal Rank: {rank}\n"
        results_str += f"Title: {repo['title']}\n"
        results_str += f"Link: {repo['link']}\n"
        results_str += f"Stars: {repo['stars']}\n"
        results_str += f"Semantic Similarity: {repo.get('semantic_similarity', 0):.4f}\n"
        results_str += f"Cross-Encoder Score: {repo.get('cross_encoder_score', 0):.4f}\n"
        results_str += f"Activity Score: {repo.get('activity_score', 0):.2f}\n"
        results_str += f"Code Quality Score: {repo.get('code_quality_score', 0)}\n"
        results_str += f"Final Score: {repo.get('final_score', 0):.4f}\n"
        results_str += f"Combined Doc Snippet: {repo['combined_doc'][:200]}...\n"
        results_str += '-' * 80 + "\n"
    # Do not update state.final_ranked here.
    return {"final_results": results_str}
