from time import sleep

def run_deepgit_agent(prompt: str):
    yield {"title": "🧠 Query Expansion", "content": f"Expanding query: {prompt}", "status": "done"}
    sleep(1)
    yield {"title": "🔍 Semantic Retrieval", "content": "Searching GitHub repositories...", "status": "done"}
    sleep(1)
    yield {"title": "📊 Ranking Repositories", "content": "Calculating relevance scores...", "status": "done"}
    sleep(1)
    yield "✅ Here are the top 3 GitHub repos:\n1. repo-A\n2. repo-B\n3. repo-C"
