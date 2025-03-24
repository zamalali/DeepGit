import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pathlib import Path
# Load environment variable
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

# Step 1: Instantiate Groq model
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=128,
    max_retries=2,
)

# Step 2: Build the prompt
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




# Step 3: Chain model and prompt
chain = prompt | llm

# Step 4: Define a function to convert queries
def convert_to_search_tags(query: str) -> str:
    print(f"\n🧠 [convert_to_search_tags] Input Query: {query}")
    response = chain.invoke({"query": query})
    print(f"🔁 [convert_to_search_tags] Output Tags: {response.content.strip()}")
    return response.content.strip()

# Example usage
if __name__ == "__main__":
    user_query = "I am looking for repos around finetuning gemini models maoinly 1.5 flash 002"
    github_query = convert_to_search_tags(user_query)
    print("🔍 GitHub Search Query:")
    print(github_query)
