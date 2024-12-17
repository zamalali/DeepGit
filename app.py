import os
import asyncio
from pathlib import Path
from llama_parse import LlamaParse
from llama_index.llms.openai import OpenAI
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator, FilterCondition
from llama_cloud.client import LlamaCloud
from llama_index.core.prompts import PromptTemplate
from llama_cloud.types import CloudDocumentCreate
from llama_index.core.async_utils import run_jobs
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
from pydantic import BaseModel, ValidationError
import re
import streamlit as st
# Environment Fixes
# torch.set_num_threads(4)  # Limit PyTorch threads to avoid overutilization
os.environ["PYTORCH_JIT"] = "0"  # Disable JIT optimization causing the torch.classes issue
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer parallelism errors
os.environ["OPENAI_API_KEY"] = "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" # Get your API key from https://platform.openai.com/account/api-keys
os.environ["LLAMA_CLOUD_API_KEY"] = "llx-XtDBMhN3DaQkDIKGSbdFSmu77xp7WvmG0UPFssiGaiSw1QvZ" # Get your API key from https://cloud.llamaindex.ai/api-key
# Initialize LLM and Embedding Model
llm = OpenAI(model='gpt-4o-mini')
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Ensure temporary directory exists
temp_dir = "temp_files"
os.makedirs(temp_dir, exist_ok=True)

# Define Metadata class
class Metadata(BaseModel):
    domain: str
    skills: List[str]
    country: List[str]

# Helper functions
def save_uploaded_files(uploaded_files):
    saved_files = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        saved_files.append(file_path)
    return saved_files

def create_llamacloud_pipeline(pipeline_name, embedding_config, transform_config):
    client = LlamaCloud(token=os.environ["LLAMA_CLOUD_API_KEY"])
    pipeline = client.pipelines.upsert_pipeline(request={
        'name': pipeline_name,
        'transform_config': transform_config,
        'embedding_config': embedding_config
    })
    return client, pipeline

async def get_metadata(text):
    prompt_template = PromptTemplate("""
    Analyze the following text and extract the following metadata:
    - Skills: Technologies, tools, and methods mentioned.
    - Domain: Relevant professional field (e.g., NLP, Machine Learning, IT, Sales, etc.).
    - Languages: List of languages known or used (e.g., German, English).
    - Education Level: Bachelor's, Master's, etc.
    - Work Location: Countries or regions mentioned.

    Text:
    {text}
    """)

    try:
        metadata = await llm.astructured_predict(
            Metadata,
            prompt_template,
            text=text,
        )
        return metadata
    except ValidationError as e:
        print(f"Error parsing metadata: {e}")
        return Metadata(domain="", skills=[], country=[])

def parse_files_concurrently(pdf_files):
    parser = LlamaParse(result_type="markdown", num_workers=4, verbose=True)  # Increase number of workers for speed

    def parse_file(pdf_file):
        try:
            docs = parser.load_data(pdf_file)
            for doc in docs:
                doc.metadata.update({'filepath': pdf_file})
            return docs
        except Exception as e:
            print(f"Error while parsing the file '{pdf_file}': {e}")
            return []

    with ThreadPoolExecutor(max_workers=8) as executor:  # Use more threads for concurrent processing
        results = list(executor.map(parse_file, pdf_files))

    # Flatten results
    documents = [doc for docs in results if docs for doc in docs]
    return documents

async def get_document_upload(doc, llm):
    full_text = doc.text

    # Get the file path of the resume
    file_path = doc.metadata['filepath']

    # Extract metadata from the resume
    extracted_metadata = await get_metadata(full_text)

    return CloudDocumentCreate(
        text=full_text,
        metadata={
            'skills': extracted_metadata.skills,
            'country': extracted_metadata.country,
            'domain': extracted_metadata.domain,
            'file_path': os.path.basename(file_path)  # Simplify file path
        }
    )

async def upload_documents(client, pipeline, documents):
    extract_jobs = [get_document_upload(doc, llm) for doc in documents]
    documents_upload_objs = await asyncio.gather(*extract_jobs)  # Use asyncio.gather for concurrency
    client.pipelines.create_batch_pipeline_documents(pipeline.id, request=documents_upload_objs)

def compute_similarity(task_description, documents):
    task_embedding = embedding_model.encode(task_description, batch_size=8, show_progress_bar=True)
    resume_embeddings = embedding_model.encode([doc.text for doc in documents], batch_size=8, show_progress_bar=True)

    similarities = cosine_similarity([task_embedding], resume_embeddings).flatten()
    seen_files = set()
    unique_candidates = []
    for i in similarities.argsort()[-10:][::-1]:  # Get top 10 matches
        file_path = os.path.basename(documents[i].metadata.get('filepath', 'Unknown Filepath'))  # Simplify file path
        if file_path not in seen_files:
            seen_files.add(file_path)
            unique_candidates.append((documents[i], similarities[i]))
    return unique_candidates

def llm_score_resume(task_description, resume_text):
    prompt = f"""
    Based on the job description:
    {task_description}

    Provide a concise and structured analysis of why this resume is a good fit. Highlight key matching points and areas of alignment. Provide a relevance score (0-100):
    {resume_text}
    """
    response = llm.complete(prompt)

    # Debugging the response format
    print("LLM Response:", response)

    # Ensure response is text or extract text from dict response
    response_text = response.get('text', response) if isinstance(response, dict) else str(response)

    # Attempt to parse score robustly using regex
    try:
        score_match = re.search(r"\bscore[:\s]+(\d{1,3})\b", response_text, re.IGNORECASE)
        score = int(score_match.group(1)) if score_match else 50
    except (IndexError, ValueError, AttributeError):
        score = 50  # Default fallback score

    # Make the response concise and markdown-friendly
    clean_response = "\n\n".join([f"- {line.strip()}" for line in response_text.splitlines() if line.strip()])

    return clean_response, score

def combine_scores(similarity_score, llm_score, weights=(0.6, 0.4)):
    return weights[0] * similarity_score + weights[1] * (llm_score / 100)

def rank_candidates(task_description, candidates):
    ranked_candidates = []
    for candidate, similarity_score in candidates:
        detailed_response, llm_score = llm_score_resume(task_description, candidate.text)
        combined_score = combine_scores(similarity_score, llm_score)
        ranked_candidates.append((os.path.basename(candidate.metadata.get('filepath', 'Unknown Filepath')), combined_score, detailed_response))  # Simplify file path

    ranked_candidates.sort(key=lambda x: x[1], reverse=True)
    return ranked_candidates

# Streamlit App
st.title("Resume Matching Tool")
st.markdown("Upload resumes and provide a job description to find the best matches.")

uploaded_files = st.file_uploader("Upload Resumes (PDF files only)", type="pdf", accept_multiple_files=True)
task_description = st.text_area("Job Description", "Enter the job description here")

if st.button("Find Matches"):
    if uploaded_files and task_description:
        with st.spinner("Processing resumes..."):
            # Step 1: Save uploaded files
            pdf_files = save_uploaded_files(uploaded_files)

            # Step 2: Parse files concurrently
            documents = parse_files_concurrently(pdf_files)

            # Step 3: Compute similarity ranking
            initial_candidates = compute_similarity(task_description, documents)

            # Step 4: Refine ranking with LLM scoring
            final_ranked_candidates = rank_candidates(task_description, initial_candidates)

            # Step 5: Display results
            if final_ranked_candidates:
                st.markdown("### Top Matching Resumes")

                # Display top 3 matches with Markdown rendering
                st.markdown("#### Top 3 Matches")
                for i, (candidate, score, detailed_response) in enumerate(final_ranked_candidates[:3]):
                    formatted_score = f"{score * 100:.0f}%"  # Convert score to percentage
                    st.markdown(f"""
                    **{i+1}. {candidate}**  
                    - **Score**: {formatted_score}  
                    - **Analysis**:  
                      {detailed_response.replace('-', '- ').replace('\n', '\n  ')}
                    ---
                    """)

                # Display remaining resumes with simplified output
                st.markdown("#### Other Matches")
                for i, (candidate, score, _) in enumerate(final_ranked_candidates[3:], start=4):
                    formatted_score = f"{score * 100:.0f}%"  # Convert score to percentage
                    st.markdown(f"""
                    **{i}. {candidate}**  
                    - **Score**: {formatted_score}  
                    ---
                    """)
            else:
                st.warning("No matching resumes found.")
    else:
        st.error("Please upload resumes and provide a task description.")

