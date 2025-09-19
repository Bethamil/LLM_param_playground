# LLM Interactive Client Design Document

## Overview
This document outlines the architecture and UI design for a Python-based Gradio web application serving as an interactive client for demonstrating Large Language Models (LLMs). The app supports multiple providers: OpenAI, OpenRouter, and Custom OpenAI-compatible endpoints.

## Overall App Structure
The application follows a modular design for easy maintenance and extension:

- **app.py**: Main Gradio application file defining the UI layout, components, and event handlers.
- **api_client.py**: Module handling API interactions with different providers using the OpenAI Python library.
- **config.py**: Configuration module storing provider settings, model lists, and default parameters.
- **utils.py**: Utility functions for input validation, error handling, and data processing.
- **requirements.txt**: List of Python dependencies (gradio, openai, python-dotenv).
- **README.md**: Setup and usage documentation.
- **.env**: Environment file for secure storage of API keys (not committed to repository).

## UI Component Layout and Interactions
The interface uses Gradio Blocks for a responsive layout:

- **Header**: Application title "LLM Interactive Client".
- **Provider Selection**: Radio buttons for "OpenAI", "OpenRouter", "Custom".
- **Custom Provider Fields** (conditionally visible): Text inputs for Base URL and API Key.
- **Model Selection**: Dropdown populated dynamically based on selected provider.
- **Input Section**:
  - Textarea for System Message.
  - Textarea for User Prompt.
- **Parameters Section**:
  - Sliders: Temperature (0.0-2.0), Max Tokens (100-4096), Top-p (0.0-1.0), Frequency Penalty (-2.0-2.0), Presence Penalty (-2.0-2.0).
  - Checkbox for Streaming Responses.
- **Generate Button**: Triggers API call.
- **Output Section**:
  - Textarea for Model Response.
  - Error Display area for API issues.
- **Metadata Section**: Displays token usage, response time, and model details.

**Interactions**:
- Provider change updates model dropdown options.
- Generate button validates inputs, calls backend, displays response or errors.
- Streaming enabled shows progressive output updates.

All components include clear labels and tooltips for user guidance.

## Backend Logic Flow
1. User selects provider and inputs parameters.
2. On generate click: Validate inputs (e.g., API key presence, URL format).
3. Instantiate OpenAI client:
   - For OpenAI/OpenRouter: Use default base URL.
   - For Custom: Use provided base URL.
4. Set API key from input or environment variable.
5. Prepare chat messages from system and user inputs.
6. Call `client.chat.completions.create()` with all parameters.
7. Handle response:
   - If streaming: Process stream chunks, update output progressively.
   - Else: Display full response.
8. Extract metadata (tokens, time) and display.
9. Catch exceptions: Display user-friendly error messages.

## Data Flow for Dynamic Elements
- Model lists stored as dictionaries in config.py (e.g., `openai_models = ["gpt-4o-mini", "gpt-4"]`).
- Provider selection triggers update of dropdown choices from config.
- For future dynamic fetching: Add API calls to retrieve models (e.g., OpenAI's models endpoint).
- Custom provider allows manual model input if needed.

## Security Considerations
- API keys handled via environment variables (python-dotenv) or secure input fields; never logged or persisted.
- Input validation: Sanitize URLs, enforce parameter ranges.
- HTTPS enforcement for base URLs.
- No client-side storage of sensitive data.
- For production: Implement OAuth or key vault integration.

## Extension Points for Future Features
- **New Providers**: Add entries to config.py and extend api_client.py logic.
- **Additional Parameters**: Include in UI and API call mapping.
- **Plugin System**: Use Python entry points for loading custom modules.
- **Multi-turn Conversations**: Add chat history state and UI.
- **File Uploads**: Extend inputs for prompt attachments.
- **Model Comparison**: Support multiple simultaneous API calls and side-by-side outputs.