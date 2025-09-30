# config.py - Configuration module for LLM Interactive Client

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Provider settings
PROVIDERS = ["OpenAI", "OpenRouter", "Custom"]

# Model lists for each provider
OPENAI_MODELS = [
    "gpt-4o-mini",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-3.5-turbo"
]

OPENROUTER_MODELS = [
    "anthropic/claude-3-haiku",
    "anthropic/claude-3-sonnet",
    "openai/gpt-4o-mini",
    "openai/gpt-4",
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemini-pro"
]

CUSTOM_MODELS = []  # Empty for custom, as model can be specified manually

# Default parameters
DEFAULT_TEMPERATURE = 1
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TOP_P = 1.0
DEFAULT_FREQUENCY_PENALTY = 0.0
DEFAULT_PRESENCE_PENALTY = 0.0
DEFAULT_STREAMING = True

# RAG (Retrieval Augmented Generation) settings
DEFAULT_RAG_ENABLED = False
DEFAULT_VECTOR_DB_NAME = "vector_db"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_RETRIEVAL_K = 5
DEFAULT_KNOWLEDGE_BASE_PATH = "knowledge-base"
RAG_FILE_PATTERNS = ["**/*.md", "**/*.txt", "**/*.pdf"]
MAX_VISUALIZATION_VECTORS = 1000

# Embedding model settings
EMBEDDING_PROVIDERS = ["OpenAI", "Custom"]
DEFAULT_EMBEDDING_PROVIDER = "OpenAI"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CUSTOM_EMBEDDING_BASE_URL = "http://localhost:11434/v1"
DEFAULT_CUSTOM_EMBEDDING_MODEL = "nomic-embed-text"

# MCP (Model Context Protocol) settings
DEFAULT_MCP_ENABLED = False
MCP_CONFIG_FILE = "mcp-server.json"
DEFAULT_MCP_TOOL_CALL_ENABLED = False

# API base URLs
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Cache for fetched models to avoid repeated API calls
_cached_models = {}

def fetch_openai_models():
    """
    Fetch available models from OpenAI API using the OpenAI client.
    
    Returns:
        list: List of available OpenAI model names, or fallback list if API call fails.
    """
    if 'openai' in _cached_models:
        return _cached_models['openai']
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return OPENAI_MODELS  # Fallback to hardcoded list
    
    try:
        client = OpenAI(api_key=api_key)
        models_response = client.models.list()
        
        # Filter for chat completion models and sort by name
        models = [model.id for model in models_response.data 
                 if model.id.startswith(('gpt-', 'text-davinci', 'text-curie', 'text-babbage', 'text-ada'))]
        models.sort()
        _cached_models['openai'] = models
        return models
    except Exception:
        return OPENAI_MODELS  # Fallback to hardcoded list

def fetch_openrouter_models():
    """
    Fetch available models from OpenRouter API using the OpenAI client.
    
    Returns:
        list: List of available OpenRouter model names, or fallback list if API call fails.
    """
    if 'openrouter' in _cached_models:
        return _cached_models['openrouter']
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return OPENROUTER_MODELS  # Fallback to hardcoded list
    
    try:
        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
        )
        models_response = client.models.list()
        
        # Extract model IDs and sort
        models = [model.id for model in models_response.data]
        models.sort()
        _cached_models['openrouter'] = models
        return models
    except Exception:
        return OPENROUTER_MODELS  # Fallback to hardcoded list

def get_models_for_provider(provider):
    """
    Get models for a specific provider, using dynamic fetching with fallback.
    
    Args:
        provider (str): The provider name ("OpenAI", "OpenRouter", "Custom")
    
    Returns:
        list: List of available models for the provider.
    """
    if provider == "OpenAI":
        return fetch_openai_models()
    elif provider == "OpenRouter":
        return fetch_openrouter_models()
    else:  # Custom
        return []

def clear_model_cache():
    """Clear the cached models to force a refresh on next fetch."""
    global _cached_models
    _cached_models = {}