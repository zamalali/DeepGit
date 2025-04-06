import os
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

# Step 1: Instantiate the Groq model with appropriate settings.
llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0.3,
    max_tokens=512,
    max_retries=3,
)

# Step 2: Build the prompt with enhanced instructions for iterative thinking and target language detection.
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are a GitHub search optimization expert.

Your job is to:
1. Read a user's query about tools, research, or tasks.
2. Detect if the query mentions a specific programming language other than Python (for example, JavaScript or JS). If so, record that language as the target language.
3. Think iteratively and generate your internal chain-of-thought enclosed in <think> ... </think> tags.
4. After your internal reasoning, output up to five GitHub-style search tags or library names that maximize repository discovery.
   Use as many tags as necessary based on the query's complexity, but never more than five.
5. If you detected a non-Python target language, append an additional tag at the end in the format target-[language] (e.g., target-javascript).
   If no specific language is mentioned, do not include any target tag.
   
Output Format:
tag1:tag2[:tag3[:tag4[:tag5[:target-language]]]]

Rules:
- Use lowercase and hyphenated keywords (e.g., image-augmentation, chain-of-thought).
- Use terms commonly found in GitHub repo names, topics, or descriptions.
- Avoid generic terms like "python", "ai", "tool", "project".
- Do NOT use full phrases or vague words like "no-code", "framework", or "approach".
- Prefer real tools, popular methods, or dataset names when mentioned.
- If your output does not strictly match the required format, correct it after your internal reasoning.
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

Input: "Find repositories implementing data augmentation pipelines in JavaScript"
Output: data-augmentation:target-javascript

Output must be ONLY the search tags separated by colons. Do not include any extra text, bullet points, or explanations.
"""),
    ("human", "{query}")
])

# Step 3: Chain the prompt with the LLM.
chain = prompt | llm

# Step 4: Define a function to parse the final search tags from the model's response.
def parse_search_tags(response: str) -> str:
    """
    Removes any internal commentary enclosed in <think> ... </think> tags
    and returns only the final searchable tags.
    """
    if "<think>" in response and "</think>" in response:
        end_index = response.index("</think>") + len("</think>")
        tags = response[end_index:].strip()
        return tags
    else:
        return response.strip()

# Step 5: Helper function to validate the output tags format using regex.
def valid_tags(tags: str) -> bool:
    """
    Validates that the output is one to six colon-separated tokens composed of lowercase letters, numbers, and hyphens.
    This allows up to five search tags and optionally one target tag.
    """
    pattern = r'^[a-z0-9-]+(?::[a-z0-9-]+){0,5}$'
    return re.match(pattern, tags) is not None

# Step 6: Define an iterative conversion function that refines the output if needed.
def iterative_convert_to_search_tags(query: str, max_iterations: int = 2) -> str:
    print(f"\n[iterative_convert_to_search_tags] Input Query: {query}")
    refined_query = query
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration+1}")
        response = chain.invoke({"query": refined_query})
        full_output = response.content.strip()
        tags_output = parse_search_tags(full_output)
        print(f"Output Tags: {tags_output}")
        if valid_tags(tags_output):
            print("Valid tags format detected.")
            return tags_output
        else:
            print("Invalid tags format. Requesting refinement...")
            refined_query = f"{query}\nPlease refine your answer so that the output strictly matches the format: tag1:tag2[:tag3[:tag4[:tag5[:target-language]]]]."
    print("Final output (may be invalid):", tags_output)
    return tags_output

# Example usage
if __name__ == "__main__":
    # Example queries for testing:
    example_queries = [
        "I am looking for repositories for data augmentation pipelines for fine-tuning LLMs",  # Default (Python)
        "Find repositories implementing data augmentation pipelines in JavaScript",          # Should return target-javascript
        "Searching for tools for instruction-based finetuning for LLaMA 2",                   # Default (Python)
        "Looking for open-source libraries for object detection using YOLO",                 # Default (Python)
        "Repos implementing chatbots in JavaScript with self-reflection capabilities"         # Should return target-javascript
    ]
    
    for q in example_queries:
        github_query = iterative_convert_to_search_tags(q)
        print("\nGitHub Search Query:")
        print(github_query)
