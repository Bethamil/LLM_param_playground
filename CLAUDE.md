# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based Gradio web application that serves as an interactive playground for Large Language Models (LLMs). It provides a user-friendly interface for experimenting with different LLM providers (OpenAI, OpenRouter, and custom OpenAI-compatible endpoints) with configurable parameters.

## Development Commands

### Running the Application
```bash
python app.py
```
The application will start a Gradio web interface accessible at `http://127.0.0.1:7860`.

### Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration
Copy `.env.example` to `.env` and configure API keys:
- `OPENAI_API_KEY`: Required for OpenAI provider
- `OPENROUTER_API_KEY`: Required for OpenRouter provider

## Architecture

### Core Files
- **app.py**: Main Gradio application with UI components and event handlers
- **config.py**: Configuration module containing provider settings, model lists, default parameters, and dynamic model fetching functions
- **requirements.txt**: Python dependencies (gradio, openai, python-dotenv)

### Key Architecture Patterns

**Provider System**: The application uses a modular provider system where each LLM provider (OpenAI, OpenRouter, Custom) has its own configuration and model fetching logic in `config.py`.

**Dynamic Model Fetching**: Models are fetched dynamically from provider APIs using cached requests with fallback to hardcoded lists if API calls fail. The caching system (`_cached_models`) prevents repeated API calls.

**OpenAI Client Pattern**: All providers use the OpenAI Python library as a unified interface, even for non-OpenAI providers like OpenRouter, by configuring custom base URLs.

**Streaming Support**: The application supports both streaming and non-streaming responses for real-time output display.

### Configuration Management
- Provider settings and model lists are centralized in `config.py`
- API keys are handled through environment variables for security
- Default parameters are defined as constants (temperature, max tokens, etc.)
- Model caching prevents repeated API calls during a session

### Error Handling
The application includes comprehensive error handling for:
- Authentication errors (invalid API keys)
- Rate limiting
- Network connectivity issues
- API errors and provider-specific issues
- Invalid model selections

### UI Architecture
- Uses Gradio Blocks for responsive layout
- Conditional visibility for custom provider fields
- Real-time parameter adjustment with sliders
- Progressive response display for streaming
- Metadata display for response time and token usage