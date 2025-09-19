# LLM Param Playground

A Python-based web application built with Gradio that serves as an interactive playground for Large Language Models (LLMs). It provides an easy-to-use interface for experimenting with LLMs from multiple providers, including OpenAI, OpenRouter, and custom OpenAI-compatible endpoints, with a focus on parameter tuning and exploration.

## Description

LLM Param Playground bridges the gap between complex LLM APIs and end-users, making AI interaction more accessible, customizable, and educational. It serves as both a demonstration tool and a practical interface for working with various LLM providers, allowing users to experiment with different models, parameters, and response modes.

## Features

- **Provider Selection**: Choose between OpenAI, OpenRouter, or Custom providers via radio buttons.
- **Dynamic Model Selection**: Dropdown menus populated based on the selected provider, with fallback to hardcoded lists if API fetching fails.
- **Configurable Parameters**: Adjust model parameters including:
  - Temperature (0.0-2.0)
  - Max tokens (100-4096)
  - Top-p (0.0-1.0)
  - Frequency penalty (-2.0-2.0)
  - Presence penalty (-2.0-2.0)
- **Response Modes**: Enable/disable streaming responses for real-time output.
- **Input Handling**: Text areas for system messages and user prompts.
- **API Integration**: Uses the OpenAI Python library to handle API calls, with support for custom endpoints.
- **Error Handling**: Comprehensive exception handling for authentication, rate limits, network issues, and API errors.
- **Metadata Display**: Shows response time, token usage, model name, and provider information.
- **Modular Architecture**: Separates UI logic (app.py), configuration (config.py), and design documentation (design.md).
- **Dynamic Model Fetching**: Attempts to fetch current models from provider APIs, with graceful fallback.
- **Secure API Key Handling**: Prioritizes environment variables over user input, with no logging or persistence of sensitive data.
- **Responsive UI**: Conditional visibility for custom provider fields, clear labels, and tooltips.
- **Streaming Support**: Progressive response display for better user experience.
- **Performance Monitoring**: Displays metadata for response time and token usage.
- **Extensibility**: Designed for future features like multi-turn conversations, file uploads, and model comparison.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/llm-param-playground.git
   cd llm-param-playground
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables (see Configuration section below).

## Usage

1. Ensure your API keys are configured (see Configuration).
2. Run the application:
   ```bash
   python app.py
   ```
3. Open your web browser and navigate to the provided local URL (typically `http://127.0.0.1:7860`).
4. Select a provider, choose a model, configure parameters, enter your prompt, and interact with the LLM.

## Configuration

Create a `.env` file in the project root based on `.env.example`. Configure the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key (required for OpenAI provider).
- `OPENROUTER_API_KEY`: Your OpenRouter API key (required for OpenRouter provider).
- `CUSTOM_BASE_URL`: Base URL for custom OpenAI-compatible endpoints (optional).
- `CUSTOM_API_KEY`: API key for custom endpoints (optional).

The application prioritizes environment variables for security and does not log or persist sensitive data.

## Goals

- Demonstrate LLM capabilities through an accessible web interface.
- Support multiple providers for flexibility and choice.
- Allow fine-tuning of model parameters for different use cases (creative vs. focused responses).
- Provide real-time feedback through streaming and robust error handling.
- Enable easy extension for new providers and features.
- Maintain security through environment variable-based API key management.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.