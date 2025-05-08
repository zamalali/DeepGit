import os
from enum import Enum
from typing import Optional
from langchain_openai import ChatOpenAI as OpenAIChat
from langchain_groq import ChatGroq as GroqChat
from langchain_core.language_models import BaseChatModel
import getpass
from pathlib import Path
from dotenv import load_dotenv

class LLMProvider(str, Enum):
    OPENAI = "openai"
    GROQ = "groq"

class LLMConfig:
    def __init__(self, provider: LLMProvider = LLMProvider.OPENAI):
        self.provider = provider
        self._load_env()
        self._validate_api_keys()
        
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
    
    def get_llm(self, 
                model_name: Optional[str] = None,
                temperature: float = 0.3,
                max_tokens: int = 512,
                max_retries: int = 2) -> BaseChatModel:
        """
        Get the configured LLM instance with the specified parameters.
        
        Args:
            model_name: Optional model name override
            temperature: Model temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in response
            max_retries: Number of retries on failure
            
        Returns:
            Configured LLM instance
        """
        if self.provider == LLMProvider.OPENAI:
            return OpenAIChat(
                model=model_name or "gpt-3.5-turbo",
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries
            )
        elif self.provider == LLMProvider.GROQ:
            return GroqChat(
                model=model_name or "llama-3.1-8b-instant",
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

# Default configuration
default_config = LLMConfig() 