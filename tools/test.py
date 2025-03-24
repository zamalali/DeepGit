import os
import requests
import subprocess
import tempfile
import shutil
import stat
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from git import Repo

# Load environment variable
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

# ---------------------------
# Step 1: Instantiate Groq model
# ---------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=128,
    max_retries=2,
)

# ---------------------------
# Step 2: Build the prompt
# ---------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are a GitHub search optimization expert.

Your job is to:
1. Read a user's query about tools, research, or tasks.
2. Return **exactly two GitHub-style search tags or library names** that maximize repository discovery.
3. Tags must represent:
   - The core task/technique (e.g., image-augmentation, instruction-tuning)
   - A specific tool, model name, or approach (e.g., albumentations, label-studio, llama2)

 Output Format:
tag-one:tag-two

 Rules:
- Use lowercase and hyphenated keywords (e.g., image-augmentation, chain-of-thought)
- Use terms commonly found in GitHub repo names, topics, or descriptions
- Avoid generic terms like "python", "ai", "tool", "project"
- Do NOT use full phrases or vague words like "no-code", "framework", "approach"
- Prefer *real tools*, *popular methods*, or *dataset names* if mentioned
- Choose high-signal keywords. Be precise.

 Excellent Examples:

Input: "No code tool to augment image and annotation"
Output: image-augmentation:albumentations

Input: "Open-source tool for labeling datasets with UI"
Output: label-studio:streamlit

Input: "Visual reasoning models trained on multi-modal datasets"
Output: multimodal-reasoning:vlm

Input: "I want repos related to instruction-based finetuning for LLaMA 2"
Output: instruction-tuning:llama2

Input: "Repos around chain of thought prompting mainly for finetuned models"
Output: chain-of-thought:finetuned-llm

Input: "I want to fine-tune Gemini 1.5 Flash model"
Output: gemini-finetuning:instruction-tuning

Input: "Need repos for document parsing with vision-language models"
Output: document-understanding:vlm

Input: "How to train custom object detection models using YOLO"
Output: object-detection:yolov5

Input: "Segment anything-like models for interactive segmentation"
Output: interactive-segmentation:segment-anything

Input: "Synthetic data generation for vision model training"
Output: synthetic-data:image-augmentation

Input: "OCR pipeline for scanned documents"
Output: ocr:document-processing

Input: "LLMs with self-reflection or reasoning chains"
Output: self-reflection:chain-of-thought

Input: "Chatbot development using open-source LLMs"
Output: chatbot:llm

Output must be ONLY two search terms separated by a colon. No extra text. No bullet points.
"""),
    ("human", "{query}")
])

# ---------------------------
# Step 3: Chain model and prompt
# ---------------------------
chain = prompt | llm

# ---------------------------
# Step 4: Define a function to convert queries
# ---------------------------
def convert_to_search_tags(query: str) -> str:
    print(f"\n🧠 [convert_to_search_tags] Input Query: {query}")
    response = chain.invoke({"query": query})
    print(f"🔁 [convert_to_search_tags] Output Tags: {response.content.strip()}")
    return response.content.strip()

# ---------------------------
# Safe File Delete for Windows
# ---------------------------
def remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

# ---------------------------
# Code Quality Checker (Robust Function)
# ---------------------------
def analyze_code_quality(repo_info):
    """
    Clone the repository and analyze Python files with flake8.
    Returns the repo_info dictionary augmented with:
      - code_quality_score: The computed quality score.
      - code_quality_issues: Total flake8 issues found.
      - python_files: Number of Python files analyzed.
    Returns None if the repo encounters errors or has no Python files.
    """
    full_name = repo_info.get('full_name', 'unknown')
    clone_url = repo_info.get('clone_url')
    temp_dir = tempfile.mkdtemp()
    repo_path = os.path.join(temp_dir, full_name.split("/")[-1])
    
    try:
        # Attempt shallow clone to save time and space
        Repo.clone_from(clone_url, repo_path, depth=1, no_single_branch=True)
        
        # Find all Python files
        py_files = list(Path(repo_path).rglob("*.py"))
        total_files = len(py_files)
        if total_files == 0:
            print(f"⚠️  No Python files found in {full_name}. Skipping repo.")
            return None
        
        # Run flake8 to collect issues
        process = subprocess.run(
            ["flake8", "--max-line-length=120", repo_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        output = process.stdout.strip()
        error_count = len(output.splitlines()) if output else 0
        issues_per_file = error_count / total_files

        # Robust scoring logic based on issues per file
        if issues_per_file <= 2:
            score = 95 + (2 - issues_per_file) * 2.5  # Range: 95–100
        elif issues_per_file <= 5:
            score = 70 + (5 - issues_per_file) * 6.5   # Range: 70–89
        elif issues_per_file <= 10:
            score = 40 + (10 - issues_per_file) * 3    # Range: 40–69
        else:
            score = max(10, 40 - (issues_per_file - 10) * 2)

        repo_info["code_quality_score"] = round(score)
        repo_info["code_quality_issues"] = error_count
        repo_info["python_files"] = total_files
        return repo_info

    except Exception as e:
        print(f"❌ Error analyzing {full_name}: {e}. Skipping repo.")
        return None
    finally:
        try:
            shutil.rmtree(temp_dir, onerror=remove_readonly)
        except Exception as cleanup_e:
            print(f"⚠️  Cleanup error for {full_name}: {cleanup_e}")

# ---------------------------
# Example usage (if run directly)
# ---------------------------
if __name__ == "__main__":
    # Test the search tag conversion
    user_query = "I am looking for repos around finetuning gemini models mainly 1.5 flash 002"
    github_query = convert_to_search_tags(user_query)
    print("🔍 GitHub Search Query:")
    print(github_query)
