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
import os
import time
import json
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

def update_judge_visibility(enable_judge):
    """
    Show judge configuration and evaluation sections only when judge is enabled.
    Also show/hide the "use same LLM" checkbox.
    """
    return (
        gr.update(visible=enable_judge),  # use_same_llm checkbox
        gr.update(visible=enable_judge),  # judge_config_section
        gr.update(visible=enable_judge)   # judge_column (output)
    )

def update_judge_llm_config_visibility(enable_judge, use_same_llm):
    """
    Show judge LLM configuration only when judge is enabled and not using same LLM.
    """
    show_config = enable_judge and not use_same_llm
    return gr.update(visible=show_config)

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

def format_metadata(model, provider, response_time, usage, error_message, judge_enabled=False, judge_data=None):
    """
    Format the response metadata for display, including judge information if available.

    Args:
        model (str): The model name.
        provider (str): The provider name.
        response_time (float): Time taken for the response.
        usage: Token usage object from the API.
        error_message (str): Error message if any.
        judge_enabled (bool): Whether judge evaluation was enabled.
        judge_data (dict): Judge metadata including model, provider, evaluation time.

    Returns:
        str: Formatted metadata string.
    """
    if error_message:
        return ""

    metadata_str = f"""
**Main Response:**
- **Model:** {model}
- **Provider:** {provider}
- **Response Time:** {response_time:.2f} seconds
- **Token Usage:** {usage.total_tokens if usage else 'N/A'}
"""

    if judge_enabled and judge_data:
        judge_model = judge_data.get('model', 'N/A')
        judge_provider = judge_data.get('provider', 'N/A')
        judge_time = judge_data.get('evaluation_time', 0)
        judge_same_llm = judge_data.get('same_llm', False)
        judge_usage = judge_data.get('usage', None)

        metadata_str += f"""
**Judge Evaluation:**
- **Judge Model:** {judge_model}
- **Judge Provider:** {judge_provider}
- **Same LLM as Main:** {'Yes' if judge_same_llm else 'No'}
- **Judge Evaluation Time:** {judge_time:.2f} seconds
- **Judge Token Usage:** {judge_usage.total_tokens if judge_usage else 'N/A'}
"""
    elif judge_enabled:
        metadata_str += f"""
**Judge Evaluation:** Enabled (evaluation in progress or failed)
"""

    return metadata_str

def evaluate_with_judge(client, judge_model, system, prompt, response, criteria, scale, temperature):
    """
    Evaluate a response using an LLM judge.

    Args:
        client (OpenAI): The judge client instance.
        judge_model (str): The judge model name.
        system (str): Original system message.
        prompt (str): Original user prompt.
        response (str): The response to evaluate.
        criteria (str): Evaluation criteria.
        scale (str): Scoring scale ("1-5", "1-10", "1-100").
        temperature (float): Judge temperature.

    Returns:
        tuple: (score, confidence, feedback, reasoning, usage)
    """
    max_score = int(scale.split('-')[1])

    judge_prompt = f"""You are an expert judge evaluating AI responses. Please evaluate the following response based on these criteria: {criteria}

Original Query: {prompt}
Original System: {system}

Response to evaluate: {response}

Please provide your evaluation in the following JSON format:
{{
    "score": <numerical score between 1 and {max_score}>,
    "confidence": <confidence in your score between 0 and 1>,
    "feedback": "<brief feedback on the response quality>",
    "reasoning": "<your reasoning process for the score>"
}}

Be strict but fair in your evaluation. Consider accuracy, helpfulness, clarity, and relevance."""

    try:
        judge_response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=temperature,
            max_completion_tokens=1000,
            stream=False
        )

        # Parse the JSON response
        result_text = judge_response.choices[0].message.content.strip()
        usage = judge_response.usage

        # Try to extract JSON from the response
        try:
            # Look for JSON in the response
            start_idx = result_text.find('{')
            end_idx = result_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = result_text[start_idx:end_idx]
                result = json.loads(json_str)

                score = float(result.get('score', 0))
                confidence = float(result.get('confidence', 0))
                feedback = str(result.get('feedback', ''))
                reasoning = str(result.get('reasoning', ''))

                return score, confidence, feedback, reasoning, usage
            else:
                # Fallback parsing if no JSON found
                return 0, 0, "Could not parse judge response", result_text, usage
        except json.JSONDecodeError:
            return 0, 0, "Invalid JSON in judge response", result_text, usage

    except Exception as e:
        return 0, 0, f"Judge evaluation failed: {str(e)}", str(e), None

def get_judge_client(judge_provider, judge_base_url, judge_api_key):
    """
    Create a judge client instance.

    Args:
        judge_provider (str): The judge provider.
        judge_base_url (str): Custom judge base URL.
        judge_api_key (str): Judge API key.

    Returns:
        OpenAI: Judge client instance.
    """
    api_base_url = get_api_base_url(judge_provider, judge_base_url)
    api_key_final = get_api_key(judge_provider, judge_api_key)
    return create_client(api_base_url, api_key_final)

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

    # Judge Configuration Section
    gr.Markdown("## LLM Judge Configuration")
    enable_judge = gr.Checkbox(value=False, label="Enable LLM Judge", info="Use another LLM to evaluate and score the response")
    use_same_llm = gr.Checkbox(value=True, label="Use same LLM as main response", info="Use the same provider and model for judging", visible=False)

    # Judge configuration (visible only when judge is enabled)
    with gr.Column(visible=False) as judge_config_section:
        # Judge provider/model configuration (hidden when using same LLM)
        with gr.Column(visible=False) as judge_llm_config:
            with gr.Row():
                judge_provider = gr.Radio(
                    choices=config.PROVIDERS,
                    value="OpenAI",
                    label="Judge Provider",
                    info="Provider for the judge LLM"
                )

            # Judge Custom Provider Configuration (visible only when Custom is selected)
            with gr.Column(visible=False) as judge_custom_row:
                gr.Markdown("### Judge Custom Provider Configuration")
                with gr.Row():
                    judge_base_url = gr.Textbox(
                        label="Judge Endpoint URL",
                        placeholder="https://your-judge-api-endpoint.com/v1",
                        info="Enter your custom judge API endpoint"
                    )
                    judge_api_key = gr.Textbox(
                        label="Judge API Key",
                        type="password",
                        info="API key for your judge endpoint"
                    )

            # Judge Model Selection
            with gr.Row():
                judge_model_dropdown = gr.Dropdown(
                    choices=config.get_models_for_provider("OpenAI"),
                    value=config.get_models_for_provider("OpenAI")[0] if config.get_models_for_provider("OpenAI") else None,
                    label="Judge Model",
                    visible=True
                )
                judge_model_textbox = gr.Textbox(
                    label="Custom Judge Model",
                    placeholder="e.g., gpt-4, claude-3-opus",
                    info="Enter the exact judge model name",
                    visible=False
                )

        # Judge Parameters (always shown when judge is enabled)
        with gr.Row():
            judge_temperature = gr.Slider(minimum=0.0, maximum=2.0, value=0.1, label="Judge Temperature", info="Lower values make judge scoring more consistent")
            scoring_scale = gr.Radio(
                choices=["1-5", "1-10", "1-100"],
                value="1-10",
                label="Scoring Scale",
                info="Scale for judge scoring"
            )

        # Judge Criteria
        judge_criteria = gr.Textbox(
            label="Judge Criteria",
            lines=3,
            value="Evaluate the response for: accuracy, helpfulness, clarity, and relevance to the query.",
            info="Criteria for the judge to evaluate the response against"
        )

    # Event handlers for judge provider selection
    judge_provider.change(
        fn=update_model_choices,
        inputs=judge_provider,
        outputs=[judge_model_dropdown, judge_model_textbox]
    )
    judge_provider.change(
        fn=update_custom_fields,
        inputs=judge_provider,
        outputs=[judge_custom_row]
    )

    # Generate button to trigger API call
    generate_btn = gr.Button("Generate")

    # Output Section
    gr.Markdown("## Output Section")
    # Textbox for displaying reasoning tokens when available
    reasoning_output = gr.Textbox(label="Model Reasoning", lines=5, visible=True)
    # Textbox for displaying successful model responses
    response_output = gr.Textbox(label="Model Response", lines=10)
    # Textbox for displaying errors (hidden by default)
    error_output = gr.Textbox(label="Errors", lines=5, visible=False)

    # Messages Accordion Section
    with gr.Accordion("Full Message Array", open=False):
        messages_output = gr.JSON(label="Conversation Messages", value=[])

    # Judge Output Section
    with gr.Column(visible=False) as judge_column:
        gr.Markdown("## Judge Evaluation")
        with gr.Row():
            judge_score = gr.Number(label="Judge Score", value=None, precision=2, info="Score from the LLM judge")
            judge_confidence = gr.Number(label="Judge Confidence", value=None, precision=2, info="Judge's confidence in the score (0-1)")
        judge_feedback = gr.Textbox(label="Judge Feedback", lines=5, info="Detailed feedback from the judge")
        judge_reasoning = gr.Textbox(label="Judge Reasoning", lines=3, info="Judge's reasoning process")

    # Event handler for judge enable/disable
    enable_judge.change(
        fn=update_judge_visibility,
        inputs=enable_judge,
        outputs=[use_same_llm, judge_config_section, judge_column]
    )

    # Event handler for use same LLM checkbox
    use_same_llm.change(
        fn=update_judge_llm_config_visibility,
        inputs=[enable_judge, use_same_llm],
        outputs=judge_llm_config
    )

    # Metadata Section (moved to bottom)
    gr.Markdown("## Metadata")
    # Markdown display for response metadata (model, time, tokens, judge info)
    metadata = gr.Markdown(label="Response Metadata")

    # Function for generating responses with error handling
    def generate_response(provider, model_dd, model_tb, base_url, api_key, system, prompt, temp, max_tok, top_p_val, freq_pen, pres_pen, stream,
                         enable_judge, use_same_llm, judge_provider, judge_base_url, judge_api_key, judge_model_dd, judge_model_tb,
                         judge_temp, criteria, scale):
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
            tuple: (gr.update for reasoning, gr.update for response, gr.update for error, metadata_str, api_messages)
        """
        # Initialize variables
        full_response = ""
        reasoning = ""
        usage = None
        error_message = ""

        # Judge evaluation variables
        judge_score = None
        judge_confidence = None
        judge_feedback = ""
        judge_reasoning = ""
        judge_data = None

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

                    # Show loading states for judge fields during streaming if judge is enabled
                    if enable_judge:
                        judge_score_value = None  # Number fields must remain None during loading
                        judge_confidence_value = None
                        judge_feedback_value = "⌛ Judge will evaluate once response is complete..."
                        judge_reasoning_value = "📝 Waiting for main response..."
                    else:
                        judge_score_value = None
                        judge_confidence_value = None
                        judge_feedback_value = ""
                        judge_reasoning_value = ""

                    yield (gr.update(value=reasoning_value),
                           gr.update(value=partial),
                           gr.update(value="", visible=False),
                           "",
                           messages,
                           gr.update(value=judge_score_value),
                           gr.update(value=judge_confidence_value),
                           gr.update(value=judge_feedback_value),
                           gr.update(value=judge_reasoning_value),
                           gr.update(visible=enable_judge))
                full_response = partial
                reasoning = last_reasoning
                usage = last_usage
            else:
                # Handle non-streaming response
                full_response, usage, reasoning = handle_non_streaming_response(client, model, messages, temp, max_tok, top_p_val, freq_pen, pres_pen)

                # For non-streaming, yield immediately with response and loading states for judge if enabled
                if enable_judge:
                    reasoning_value = reasoning if reasoning else "No reasoning tokens included"
                    response_time = time.time() - start_time
                    metadata = format_metadata(model, provider, response_time, usage, "", judge_enabled=False)

                    yield (gr.update(value=reasoning_value),
                           gr.update(value=full_response),
                           gr.update(value="", visible=False),
                           metadata,
                           prepare_messages(system, prompt) + [{"role": "assistant", "content": full_response}],
                           gr.update(value=None),  # Number fields must remain None during loading
                           gr.update(value=None),
                           gr.update(value="🤖 Judge is evaluating the response..."),
                           gr.update(value="⏳ Processing evaluation criteria..."),
                           gr.update(visible=enable_judge))

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

        # Add assistant response to messages array if successful
        if full_response and not error_message:
            messages.append({"role": "assistant", "content": full_response})

            # Show loading states for judge evaluation if enabled
            if enable_judge:
                reasoning_value = reasoning if reasoning else "No reasoning tokens included"
                metadata = format_metadata(model, provider, response_time, usage, error_message, judge_enabled=False)

                # Yield with loading states for judge fields
                yield (gr.update(value=reasoning_value),
                       gr.update(value=full_response),
                       gr.update(value="", visible=False),
                       metadata,
                       messages,
                       gr.update(value=None),  # Number fields must remain None during loading
                       gr.update(value=None),
                       gr.update(value="🤖 Judge is evaluating the response..."),
                       gr.update(value="⏳ Processing evaluation criteria..."),
                       gr.update(visible=enable_judge))

            # Perform judge evaluation if enabled
            if enable_judge:
                judge_start_time = time.time()
                try:
                    # Use same LLM settings or judge-specific settings
                    if use_same_llm:
                        judge_model = model
                        judge_client = client
                        judge_provider_name = provider
                    else:
                        judge_model = get_model(judge_provider, judge_model_dd, judge_model_tb)
                        judge_client = get_judge_client(judge_provider, judge_base_url, judge_api_key)
                        judge_provider_name = judge_provider

                    judge_score, judge_confidence, judge_feedback, judge_reasoning, judge_usage = evaluate_with_judge(
                        judge_client, judge_model, system, prompt, full_response, criteria, scale, judge_temp
                    )

                    judge_evaluation_time = time.time() - judge_start_time
                    judge_data = {
                        'model': judge_model,
                        'provider': judge_provider_name,
                        'evaluation_time': judge_evaluation_time,
                        'same_llm': use_same_llm,
                        'usage': judge_usage
                    }
                except Exception as e:
                    judge_score = None
                    judge_confidence = None
                    judge_feedback = f"Judge evaluation failed: {str(e)}"
                    judge_reasoning = str(e)
                    judge_evaluation_time = time.time() - judge_start_time
                    judge_data = {
                        'model': 'N/A',
                        'provider': 'N/A',
                        'evaluation_time': judge_evaluation_time,
                        'same_llm': use_same_llm,
                        'usage': None
                    }

        # Format metadata for display
        if enable_judge and judge_data is not None:
            metadata = format_metadata(model, provider, response_time, usage, error_message, judge_enabled=True, judge_data=judge_data)
        else:
            metadata = format_metadata(model, provider, response_time, usage, error_message, judge_enabled=False)

        # Yield with appropriate visibility for error output
        if error_message:
            yield (gr.update(value="No reasoning tokens included"),
                   gr.update(value=""),
                   gr.update(value=error_message, visible=True),
                   metadata,
                   messages,
                   gr.update(value=judge_score),
                   gr.update(value=judge_confidence),
                   gr.update(value=judge_feedback),
                   gr.update(value=judge_reasoning),
                   gr.update(visible=enable_judge))
        else:
            reasoning_value = reasoning if reasoning else "No reasoning tokens included"
            yield (gr.update(value=reasoning_value),
                   gr.update(value=full_response),
                   gr.update(value="", visible=False),
                   metadata,
                   messages,
                   gr.update(value=judge_score),
                   gr.update(value=judge_confidence),
                   gr.update(value=judge_feedback),
                   gr.update(value=judge_reasoning),
                   gr.update(visible=enable_judge))

    # Event handler for the generate button
    # Calls generate_response function with all input values and updates output components
    generate_btn.click(
        fn=generate_response,
        inputs=[provider_radio, model_dropdown, model_textbox, custom_base_url, custom_api_key, system_message, user_prompt, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, streaming,
                enable_judge, use_same_llm, judge_provider, judge_base_url, judge_api_key, judge_model_dropdown, judge_model_textbox,
                judge_temperature, judge_criteria, scoring_scale],
        outputs=[reasoning_output, response_output, error_output, metadata, messages_output, judge_score, judge_confidence, judge_feedback, judge_reasoning, judge_column]
    )

# Main execution block
# Launches the Gradio application when the script is run directly
if __name__ == "__main__":
    demo.launch()
