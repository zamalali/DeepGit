import os
from enum import Enum
from typing import Optional
from langchain_openai import ChatOpenAI as OpenAIChat
from langchain_groq import ChatGroq as GroqChat
from langchain_core.language_models import BaseChatModel
import getpass
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

class LLMProvider(str, Enum):
    OPENAI = "openai"
    GROQ = "groq"

class PipelineType(str, Enum):
    GROQ_PIPELINE = "groq_pipeline"  # Groq LLM + ColBERT + SentenceTransformer
    OPENAI_PIPELINE = "openai_pipeline"  # OpenAI LLM + OpenAI Embeddings + OpenAI Reranking

class LLMConfig(BaseModel):
    provider: LLMProvider = Field(default=LLMProvider.OPENAI)
    pipeline_type: PipelineType = Field(default=PipelineType.OPENAI_PIPELINE)
    model: str = Field(default="gpt-3.5-turbo")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=1000)

    def _load_env(self):
        """Load environment variables from .env file"""
        dotenv_path = Path(__file__).resolve().parent.parent / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path)
    
    def _validate_api_keys(self):
        """Validate and request API keys if not present"""
        if self.provider == LLMProvider.OPENAI:
            if "OPENAI_API_KEY" not in os.environ:
                os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
        elif self.provider == LLMProvider.GROQ:
            if "GROQ_API_KEY" not in os.environ:
                os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
    
    def get_llm(self, **kwargs):
        """
        Get the configured LLM instance with the specified parameters.
        
        Args:
            model_name: Optional model name override
            temperature: Model temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            
        Returns:
            Configured LLM instance
        """
        if self.provider == LLMProvider.GROQ:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model_name=self.model,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens)
            )
        elif self.provider == LLMProvider.OPENAI:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model_name=self.model,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens)
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

# Default configuration
default_config = LLMConfig() 