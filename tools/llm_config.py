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

class GroqModelType(str, Enum):
    DEEPSEEK = "deepseek-r1-distill-llama-70b"  # For main chat and decision making
    LLAMA_INSTANT = "llama-3.1-8b-instant"      # For quick responses and evaluation
    LLAMA_7B = "llama2-7b-4096"                 # For general purpose tasks

class LLMConfig(BaseModel):
    provider: LLMProvider = Field(default=LLMProvider.OPENAI)
    pipeline_type: PipelineType = Field(default=PipelineType.OPENAI_PIPELINE)
    model: str = Field(default="gpt-3.5-turbo")
    groq_model: GroqModelType = Field(default=GroqModelType.DEEPSEEK)
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=1000)
    max_retries: int = Field(default=2)
    timeout: Optional[int] = Field(default=None)

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
            max_retries: Number of retries on failure
            timeout: Request timeout in seconds
            
        Returns:
            Configured LLM instance
        """
        if self.provider == LLMProvider.GROQ:
            from langchain_groq import ChatGroq
            # Use Groq-specific model names
            model_name = kwargs.get("model_name", self.groq_model.value)
            return ChatGroq(
                model_name=model_name,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                max_retries=kwargs.get("max_retries", self.max_retries),
                timeout=kwargs.get("timeout", self.timeout)
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

    @property
    def default_model(self) -> str:
        """Get the default model name for the current provider"""
        if self.provider == LLMProvider.GROQ:
            return self.groq_model.value
        else:
            return "gpt-3.5-turbo"

    def get_evaluation_llm(self) -> BaseChatModel:
        """Get LLM configured for evaluation tasks"""
        return self.get_llm(
            model_name=GroqModelType.LLAMA_INSTANT.value,
            temperature=0.2,
            max_tokens=100,
            timeout=15
        )

    def get_decision_llm(self) -> BaseChatModel:
        """Get LLM configured for decision making tasks"""
        return self.get_llm(
            model_name=GroqModelType.DEEPSEEK.value,
            temperature=0.3,
            max_tokens=512,
            max_retries=2
        )

    def get_chat_llm(self) -> BaseChatModel:
        """Get LLM configured for general chat tasks"""
        return self.get_llm(
            model_name=GroqModelType.DEEPSEEK.value,
            temperature=0.3
        )

# Default configuration
default_config = LLMConfig() 