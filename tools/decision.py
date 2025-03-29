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
     """You are a minimal, resource-efficient filtering agent for a GitHub research tool.

Your job is to decide whether code-level analysis (e.g., flake8, static checks, linting) should be run on a set of repositories.
**Code analysis should almost never run** — only when the user is **explicitly and repeatedly focused on code structure, correctness, or quality**.

You must return:
- `0` → **90 percent of the time**. For nearly all queries, especially high-level, research, exploratory, or implementation-related queries.
- `1` → Only if the user uses keywords like: "clean code", "linting", "flake8", "code correctness", "static analysis", or **explicitly demands code quality checks**.

Also skip analysis if:
- The number of repositories is above 30.
- The query is about concepts, papers, models, architecture, tutorials, demos, agents, or research.
- The user does not emphasize code hygiene or correctness.

Examples:
- "Show me Gemini agents using ReAct" with 25 repos → `0`
- "Find repos with solid implementation of MoE routing" with 35 repos → `0`
- "Repos with perfect flake8 compliance" with 20 repos → `1`
- "Production-level, bug-free codebases only!" with 15 repos → `1`
- "Tutorials for dataset loaders in PyTorch" with 80 repos → `0`

Only return one digit: `0` or `1`. No comments, no formatting, no explanations.
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
