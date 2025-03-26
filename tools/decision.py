from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env variables
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

# LLM setup: DeepSeek-R1-Distill
llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0.3,
    max_tokens=512,
    max_retries=2,
)

# Prompt for decision making
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a smart decision-making agent for a GitHub research assistant.

Your job is to decide whether code-level analysis (like flake8 or static checks) should be performed on the repositories returned from a search.

You must consider:
1. **The user's goal**, described in the query.
2. **The number of repositories fetched**. If too many repos are returned, code-level analysis is too slow and unnecessary.

Output strictly `1` or `0`:
- Return `1` → if the user’s query clearly requires understanding code quality or structure (e.g., good coding practices, error-free code, implementations, model finetuning internals).
- Return `0` → if the query is high-level, conceptual, exploratory, or if the repo count is too large (e.g., >100).

Some examples:
- "Repos that implement Chain of Thought for LLMs" with 25 repos → `0`
- "Best codebases for fine-tuning LLaMA 2" with 30 repos → `1`
- "Papers with code for dataset preprocessing" with 120 repos → `0`
- "Repos that use flake8 to enforce quality" with 15 repos → `1`

Only return one number. No explanation. No formatting. No extra text.
"""),
    ("human", "Query: {query}\nRepo count: {repo_count}")
])

chain = prompt | llm

# Final function
def should_run_code_analysis(query: str, repo_count: int) -> int:
    print(f"\n[Decision Maker] Query: {query} | Repo Count: {repo_count}")
    response = chain.invoke({"query": query, "repo_count": repo_count})
    
    full_output = response.content.strip()
    print(f"\n[thinking]\n{full_output}\n")

    # Parse final line for the decision
    lines = full_output.splitlines()
    # Try to get last non-empty line
    for line in reversed(lines):
        line = line.strip()
        if line in ["0", "1"]:
            print(f"[Decision Maker] Decision: {line}")
            return int(line)

    print("[Decision Maker] Failed to extract a valid decision. Defaulting to 0.")
    return 0


# Example usage
if __name__ == "__main__":
    query = "I want to find a real quick guide on custom training yolo"
    repo_count = 34
    decision = should_run_code_analysis(query, repo_count)
    print("Should run code analysis?", decision)
