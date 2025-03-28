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
    model="deepseek-r1-distill-llama-70b",
    temperature=0.3,
    max_tokens=512,
    max_retries=3,
)

# Step 2: Build the prompt with improved sample pairs
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are a GitHub search optimization expert.

Your job is to:
1. Read a user's query about tools, research, or tasks.
2. Return up to five GitHub-style search tags or library names that maximize repository discovery.
   Use as many tags as necessary based on the query's complexity, but never more than five.
3. Tags must represent:
   - The core task or technique (e.g., image-augmentation, object-detection)
   - A specific tool, model name, or approach (e.g., albumentations, yolov5, unet)
   - Optionally, additional relevant aspects such as methodology (e.g., transformer, attention)

Output Format:
tag1:tag2[:tag3[:tag4[:tag5]]]

Rules:
- Use lowercase and hyphenated keywords (e.g., image-augmentation, chain-of-thought).
- Use terms commonly found in GitHub repo names, topics, or descriptions.
- Avoid generic terms like "python", "ai", "tool", "project".
- Do NOT use full phrases or vague words like "no-code", "framework", or "approach".
- Prefer real tools, popular methods, or dataset names when mentioned.
- Choose high-signal keywords to ensure the search yields the most relevant GitHub repositories.

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
Output: gemini-finetuning:flash002

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

Input: "Deep learning-based object detection with YOLO and transformer architecture"
Output: object-detection:yolov5:transformer

Input: "Semantic segmentation for medical images using UNet with attention mechanism"
Output: semantic-segmentation:unet:attention

Output must be ONLY the search tags separated by colons. Do not include any extra text, bullet points, or explanations.
"""),
    ("human", "{query}")
])

# Step 3: Chain model and prompt
chain = prompt | llm

# Step 4: Define a function to convert queries
def parse_search_tags(response: str) -> str:
    """
    Removes any internal commentary enclosed in <think> ... </think> tags
    and returns only the final searchable tags.
    """
    if "<think>" in response and "</think>" in response:
        end_index = response.index("</think>") + len("</think>")
        # Everything after the </think> tag is considered the search tags
        tags = response[end_index:].strip()
        return tags
    else:
        return response.strip()

# Step 5: Define a function to convert queries using the chain and parser
def convert_to_search_tags(query: str) -> str:
    print(f"\n🧠 [convert_to_search_tags] Input Query: {query}")
    response = chain.invoke({"query": query})
    full_output = response.content.strip()
    tags_output = parse_search_tags(full_output)
    print(f"🔁 [convert_to_search_tags] Output Tags: {tags_output}")
    return tags_output

# Example usage
if __name__ == "__main__":
    user_query = "I am looking for repos around finetuning gemini models mainly 1.5 flash 002"
    github_query = convert_to_search_tags(user_query)
    print("🔍 GitHub Search Query:")
    print(github_query)