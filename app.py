"""
LLM Interactive Client Application

This module provides a Gradio-based web interface for interacting with various LLM providers
including OpenAI, OpenRouter, and custom API endpoints. It supports real-time streaming responses,
configurable parameters, and comprehensive error handling.

Features:
- Multiple provider support (OpenAI, OpenRouter, Custom)
- Dynamic model selection based on provider
- Configurable generation parameters (temperature, max tokens, etc.)
- Streaming and non-streaming response modes
- Error handling with user-friendly messages
- Response metadata display

Dependencies:
- gradio: For the web interface
- openai: For API client functionality
- config: Module containing configuration constants
"""

import gradio as gr
import config
import openai
import os
import time
from openai import OpenAI, AuthenticationError, RateLimitError, APIError, NotFoundError, APIConnectionError

def update_model_choices(provider):
    """
    Update the model dropdown choices based on the selected provider.
    For Custom provider, hide dropdown and show text input.
    Uses dynamic model fetching with fallback to hardcoded lists.
    """
    if provider == "OpenAI":
        models = config.get_models_for_provider("OpenAI")
        return gr.update(choices=models, value=models[0] if models else None, visible=True), gr.update(visible=False)
    elif provider == "OpenRouter":
        models = config.get_models_for_provider("OpenRouter")
        return gr.update(choices=models, value=models[0] if models else None, visible=True), gr.update(visible=False)
    elif provider == "Custom":
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(choices=[], visible=False), gr.update(visible=False)

def update_custom_fields(provider):
    """
    Show custom provider fields only when Custom is selected.
    """
    visible = provider == "Custom"
    return gr.update(visible=visible)

def get_model(provider, model_dd, model_tb):
    """
    Determine the model name based on the selected provider.

    Args:
        provider (str): The selected provider ("OpenAI", "OpenRouter", "Custom").
        model_dd (str): Model selected from dropdown.
        model_tb (str): Model entered in textbox for custom provider.

    Returns:
        str: The model name to use for the API call.
    """
    if provider == "Custom":
        return model_tb
    else:
        return model_dd

def get_api_base_url(provider, base_url):
    """
    Get the API base URL based on the provider.

    Args:
        provider (str): The selected provider.
        base_url (str): Custom base URL provided by user.

    Returns:
        str: The base URL for the API client.
    """
    if provider == "OpenAI":
        return config.OPENAI_BASE_URL
    elif provider == "OpenRouter":
        return config.OPENROUTER_BASE_URL
    else:  # Custom
        return base_url

def get_api_key(provider, api_key):
    """
    Retrieve the API key, preferring environment variables over user input.

    Args:
        provider (str): The selected provider.
        api_key (str): API key provided by user.

    Returns:
        str or None: The API key to use for authentication, or None if no key available.
    """
    if provider == "OpenAI":
        key = os.getenv("OPENAI_API_KEY") or api_key
        return key if key else None
    elif provider == "OpenRouter":
        key = os.getenv("OPENROUTER_API_KEY") or api_key
        return key if key else None
    else:  # Custom
        return api_key if api_key else None

def create_client(api_base_url, api_key_final):
    """
    Create an OpenAI client instance with the specified base URL and API key.

    Args:
        api_base_url (str): The base URL for the API.
        api_key_final (str or None): The API key for authentication.

    Returns:
        OpenAI: An instance of the OpenAI client.

    Raises:
        ValueError: If api_key_final is None or empty.
    """
    if not api_key_final:
        raise ValueError("API key is required but not provided. Please set the API key via environment variable or input field.")
    return OpenAI(base_url=api_base_url, api_key=api_key_final)

def prepare_messages(system, prompt):
    """
    Prepare the list of messages for the API call.

    Args:
        system (str): The system message defining AI behavior.
        prompt (str): The user prompt.

    Returns:
        list: List of message dictionaries.
    """
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]

def handle_streaming_response(client, model, messages, temp, max_tok, top_p_val, freq_pen, pres_pen):
    """
    Handle streaming API response and yield partial responses with usage and reasoning.

    Args:
        client (OpenAI): The OpenAI client instance.
        model (str): The model name.
        messages (list): List of messages.
        temp (float): Temperature parameter.
        max_tok (int): Max tokens.
        top_p_val (float): Top-p parameter.
        freq_pen (float): Frequency penalty.
        pres_pen (float): Presence penalty.

    Yields:
        tuple: (partial_response, usage, reasoning)
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp,
        max_completion_tokens=max_tok,
        top_p=top_p_val,
        frequency_penalty=freq_pen,
        presence_penalty=pres_pen,
        stream=True
    )

    full_response = ""
    full_reasoning = ""
    usage = None
    for chunk in response:
        if chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content
            yield full_response, usage, full_reasoning
        if hasattr(chunk.choices[0].delta, 'reasoning') and chunk.choices[0].delta.reasoning:
            full_reasoning += chunk.choices[0].delta.reasoning
            yield full_response, usage, full_reasoning
        if chunk.usage:
            usage = chunk.usage
            yield full_response, usage, full_reasoning

def handle_non_streaming_response(client, model, messages, temp, max_tok, top_p_val, freq_pen, pres_pen):
    """
    Handle non-streaming API response and extract the response, usage, and reasoning.

    Args:
        client (OpenAI): The OpenAI client instance.
        model (str): The model name.
        messages (list): List of messages.
        temp (float): Temperature parameter.
        max_tok (int): Max tokens.
        top_p_val (float): Top-p parameter.
        freq_pen (float): Frequency penalty.
        pres_pen (float): Presence penalty.

    Returns:
        tuple: (full_response, usage, reasoning)
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp,
        max_completion_tokens=max_tok,
        top_p=top_p_val,
        frequency_penalty=freq_pen,
        presence_penalty=pres_pen,
        stream=False
    )
    full_response = response.choices[0].message.content
    usage = response.usage
    reasoning = getattr(response.choices[0].message, 'reasoning', None) or ""
    return full_response, usage, reasoning

def format_metadata(model, provider, response_time, usage, error_message):
    """
    Format the response metadata for display.

    Args:
        model (str): The model name.
        provider (str): The provider name.
        response_time (float): Time taken for the response.
        usage: Token usage object from the API.
        error_message (str): Error message if any.

    Returns:
        str: Formatted metadata string.
    """
    if error_message:
        return ""
    else:
        return f"""
**Model:** {model}
**Provider:** {provider}
**Response Time:** {response_time:.2f} seconds
**Token Usage:** {usage.total_tokens if usage else 'N/A'}
"""

# Gradio interface definition
# Creates the web UI for the LLM client using Gradio Blocks
with gr.Blocks(title="LLM Interactive Client") as demo:
    # Main title
    gr.Markdown("# LLM Interactive Client")

    # Provider Selection Section
    # Radio button to choose between OpenAI, OpenRouter, or Custom provider
    provider_radio = gr.Radio(
        choices=config.PROVIDERS,
        value="OpenAI",
        label="Select Provider",
        info="Choose the LLM provider"
    )

    # Custom Provider Configuration (visible only when Custom is selected)
    with gr.Column(visible=False) as custom_row:
        gr.Markdown("### Custom Provider Configuration")
        with gr.Row():
            custom_base_url = gr.Textbox(
                label="Custom Endpoint URL", 
                placeholder="https://your-api-endpoint.com/v1",
                info="Enter your custom OpenAI-compatible API endpoint"
            )
            custom_api_key = gr.Textbox(
                label="Custom API Key", 
                type="password",
                info="API key for your custom endpoint"
            )

    # Model Selection Section
    # Dropdown for predefined models, textbox for custom model names
    with gr.Row():
        initial_models = config.get_models_for_provider("OpenAI")
        model_dropdown = gr.Dropdown(
            choices=initial_models,
            value=initial_models[0] if initial_models else None,
            label="Select Model",
            visible=True
        )
        model_textbox = gr.Textbox(
            label="Custom Model Name",
            placeholder="e.g., gpt-3.5-turbo, claude-3-sonnet",
            info="Enter the exact model name for your custom endpoint",
            visible=False
        )

    # Event handlers for provider selection
    # Update model choices and custom fields visibility based on provider
    provider_radio.change(
        fn=update_model_choices,
        inputs=provider_radio,
        outputs=[model_dropdown, model_textbox]
    )
    provider_radio.change(
        fn=update_custom_fields,
        inputs=provider_radio,
        outputs=[custom_row]
    )

    # Input Section
    gr.Markdown("## Input Section")

    # System message input to define AI behavior
    system_message = gr.Textbox(label="System Message", lines=2, info="Set the AI's role or behavior, e.g., 'You are a helpful assistant.'")

    # User prompt input for the query
    user_prompt = gr.Textbox(label="User Prompt", lines=4, info="Enter your message or query for the AI.")

    # Parameters Section
    gr.Markdown("## Parameters")
    with gr.Row():
        temperature = gr.Slider(minimum=0.0, maximum=2.0, value=config.DEFAULT_TEMPERATURE, label="Temperature", info="Controls the randomness of the output. Higher values (e.g., 1.0) make responses more creative and varied, while lower values (e.g., 0.1) make them more focused and deterministic.")
        max_tokens = gr.Slider(minimum=100, maximum=4096, value=config.DEFAULT_MAX_TOKENS, label="Max Tokens", info="The maximum number of tokens (words or parts of words) the model can generate in its response.")
    with gr.Row():
        top_p = gr.Slider(minimum=0.0, maximum=1.0, value=config.DEFAULT_TOP_P, label="Top-p", info="Nucleus sampling parameter. Lower values focus on the most likely tokens, making output more coherent; higher values allow more diversity.")
        frequency_penalty = gr.Slider(minimum=-2.0, maximum=2.0, value=config.DEFAULT_FREQUENCY_PENALTY, label="Frequency Penalty", info="Reduces repetition of frequent tokens. Positive values decrease the likelihood of repeating words already used.")
        presence_penalty = gr.Slider(minimum=-2.0, maximum=2.0, value=config.DEFAULT_PRESENCE_PENALTY, label="Presence Penalty", info="Reduces repetition of topics. Positive values encourage introducing new topics in the response.")
    # Checkbox to enable/disable streaming responses
    streaming = gr.Checkbox(value=config.DEFAULT_STREAMING, label="Streaming Responses")

    # Generate button to trigger API call
    generate_btn = gr.Button("Generate")

    # Output Section
    gr.Markdown("## Output Section")
    # Textbox for displaying reasoning tokens when available
    reasoning_output = gr.Textbox(label="Model Reasoning", lines=5, visible=True)
    # Textbox for displaying successful model responses
    response_output = gr.Textbox(label="Model Response", lines=10)
    # Textbox for displaying errors (hidden by default)
    error_output = gr.Textbox(label="Errors", visible=False)

    # Metadata Section
    gr.Markdown("## Metadata Section")
    # Markdown display for response metadata (model, time, tokens)
    metadata = gr.Markdown(label="Response Metadata")

    # Function for generating responses with error handling
    def generate_response(provider, model_dd, model_tb, base_url, api_key, system, prompt, temp, max_tok, top_p_val, freq_pen, pres_pen, stream):
        """
        Generate response from LLM API based on selected provider and parameters.
        Includes comprehensive error handling for API calls.

        Args:
            provider (str): Selected provider ("OpenAI", "OpenRouter", "Custom")
            model_dd (str): Model from dropdown (for OpenAI/OpenRouter)
            model_tb (str): Model from textbox (for Custom)
            base_url (str): Custom base URL
            api_key (str): Custom API key
            system (str): System message
            prompt (str): User prompt
            temp (float): Temperature parameter
            max_tok (int): Max tokens
            top_p_val (float): Top-p parameter
            freq_pen (float): Frequency penalty
            pres_pen (float): Presence penalty
            stream (bool): Whether to use streaming

        Yields:
            tuple: (gr.update for response, gr.update for reasoning, gr.update for error, metadata_str)
        """
        # Initialize variables
        full_response = ""
        reasoning = ""
        usage = None
        error_message = ""

        # Determine model, API base URL, and API key using helper functions
        model = get_model(provider, model_dd, model_tb)
        api_base_url = get_api_base_url(provider, base_url)
        api_key_final = get_api_key(provider, api_key)

        # Create OpenAI client instance
        client = create_client(api_base_url, api_key_final)

        # Prepare messages for the API call
        messages = prepare_messages(system, prompt)

        # Record start time for response time calculation
        start_time = time.time()

        try:
            # Make API call based on streaming preference
            if stream:
                last_usage = None
                last_reasoning = ""
                for partial, last_usage, last_reasoning in handle_streaming_response(client, model, messages, temp, max_tok, top_p_val, freq_pen, pres_pen):
                    reasoning_value = last_reasoning if last_reasoning else "No reasoning tokens included"
                    yield (gr.update(value=reasoning_value),
                           gr.update(value=partial),
                           gr.update(value="", visible=False),
                           "")
                full_response = partial
                reasoning = last_reasoning
                usage = last_usage
            else:
                # Handle non-streaming response
                full_response, usage, reasoning = handle_non_streaming_response(client, model, messages, temp, max_tok, top_p_val, freq_pen, pres_pen)

        except ValueError as e:
            error_message = str(e)
        except AuthenticationError:
            error_message = "Error: Invalid API key. Please check your API key and try again."
        except RateLimitError:
            error_message = "Error: Rate limit exceeded. Please wait and try again later."
        except NotFoundError:
            error_message = "Error: Invalid model selected. Please choose a valid model."
        except APIConnectionError:
            error_message = "Error: Network connection issue. Please check your internet connection and try again."
        except APIError as e:
            error_message = f"Error: API error occurred - {str(e)}"
        except Exception as e:
            error_message = f"Error: Unexpected error - {str(e)}"

        # Calculate total response time
        end_time = time.time()
        response_time = end_time - start_time

        # Format metadata for display
        metadata = format_metadata(model, provider, response_time, usage, error_message)

        # Yield with appropriate visibility for error output
        if error_message:
            yield (gr.update(value="No reasoning tokens included"),
                   gr.update(value=""),
                   gr.update(value=error_message, visible=True),
                   metadata)
        else:
            reasoning_value = reasoning if reasoning else "No reasoning tokens included"
            yield (gr.update(value=reasoning_value),
                   gr.update(value=full_response),
                   gr.update(value="", visible=False),
                   metadata)

    # Event handler for the generate button
    # Calls generate_response function with all input values and updates output components
    generate_btn.click(
        fn=generate_response,
        inputs=[provider_radio, model_dropdown, model_textbox, custom_base_url, custom_api_key, system_message, user_prompt, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, streaming],
        outputs=[reasoning_output, response_output, error_output, metadata]
    )

# Main execution block
# Launches the Gradio application when the script is run directly
if __name__ == "__main__":
    demo.launch()