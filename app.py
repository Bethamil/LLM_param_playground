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
from openai import OpenAI, AuthenticationError, RateLimitError, APIError, NotFoundError, APIConnectionError
from rag import RAGManager

# Initialize RAG manager
rag_manager = RAGManager(db_name=config.DEFAULT_VECTOR_DB_NAME, openai_api_key=os.getenv("OPENAI_API_KEY"))

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

def handle_api_response(client, model, messages, temp, max_tok, top_p_val, freq_pen, pres_pen, stream=False):
    """
    Handle API response with unified streaming and non-streaming logic.

    Args:
        client (OpenAI): The OpenAI client instance.
        model (str): The model name.
        messages (list): List of messages.
        temp (float): Temperature parameter.
        max_tok (int): Max tokens.
        top_p_val (float): Top-p parameter.
        freq_pen (float): Frequency penalty.
        pres_pen (float): Presence penalty.
        stream (bool): Whether to use streaming mode.

    Yields:
        tuple: (response_content, usage, reasoning)
        - For streaming: yields partial results as they arrive
        - For non-streaming: yields final result once
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp,
        max_completion_tokens=max_tok,
        top_p=top_p_val,
        frequency_penalty=freq_pen,
        presence_penalty=pres_pen,
        stream=stream
    )

    if stream:
        # Streaming mode: yield partial results as they arrive
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
    else:
        # Non-streaming mode: yield final result once
        full_response = response.choices[0].message.content
        usage = response.usage
        reasoning = getattr(response.choices[0].message, 'reasoning', None) or ""
        yield full_response, usage, reasoning

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

# RAG-specific functions
def initialize_rag_with_api_key(provider, api_key, custom_api_key):
    """Initialize RAG manager with appropriate API key based on provider."""
    openai_key = None

    if provider == "OpenAI":
        openai_key = os.getenv("OPENAI_API_KEY") or api_key
    elif provider == "OpenRouter":
        openai_key = os.getenv("OPENROUTER_API_KEY") or api_key
    elif provider == "Custom":
        openai_key = custom_api_key

    if openai_key:
        rag_manager.set_api_key(openai_key)
        return True, "RAG initialized with API key"
    else:
        return False, "No API key available for RAG initialization"

def load_knowledge_base(knowledge_base_path, file_pattern):
    """Load documents from knowledge base."""
    if not knowledge_base_path:
        return False, "Please specify a knowledge base path", "No documents loaded"

    success, message, doc_count = rag_manager.load_documents_from_folder(knowledge_base_path, file_pattern)
    status_msg = f"Status: {message}"
    return success, status_msg, f"Documents: {doc_count}" if success else "Documents: 0"

def create_vector_db():
    """Create vector database from loaded documents."""
    success, message = rag_manager.create_vector_database()
    return success, f"Vector DB: {message}"

def load_existing_vector_db():
    """Load existing vector database."""
    success, message = rag_manager.load_existing_vector_database()
    return success, f"Vector DB: {message}"

def auto_load_rag_database(rag_enabled, provider):
    """Auto-load existing database when RAG is enabled (using environment API keys)."""
    if not rag_enabled:
        return "RAG disabled"

    # Initialize RAG with environment API key
    rag_success, rag_message = initialize_rag_with_api_key(provider, None, None)  # Use env keys
    if not rag_success:
        return f"⚠️ {rag_message}"

    # Check if database exists and try to load it
    if os.path.exists(config.DEFAULT_VECTOR_DB_NAME):
        success, message = load_existing_vector_db()
        if success:
            stats = rag_manager.get_database_stats()
            if stats.get("status") == "Active":
                return f"✅ Auto-loaded existing database | Documents: {stats['document_count']}"
            else:
                return f"⚠️ Database exists but couldn't load: {stats['status']}"
        else:
            return f"⚠️ Database folder exists but couldn't load"
    else:
        return "📂 No database found - click Initialize to create one"

def create_2d_visualization():
    """Create 2D vector visualization."""
    fig = rag_manager.create_vector_visualization(dimensions=2)
    return fig

def create_3d_visualization():
    """Create 3D vector visualization."""
    fig = rag_manager.create_vector_visualization(dimensions=3)
    return fig


def initialize_or_update_rag_database(knowledge_base_path, file_pattern):
    """
    Combined function to load documents and create/update vector database in one step.
    Only recreates embeddings when necessary (new documents or no existing database).

    Args:
        knowledge_base_path (str): Path to knowledge base folder
        file_pattern (str): File pattern to match

    Returns:
        tuple: (status_message, doc_count_message)
    """
    try:
        # Check if database already exists and is working
        if os.path.exists(config.DEFAULT_VECTOR_DB_NAME):
            try:
                existing_success, _ = load_existing_vector_db()
                if existing_success:
                    stats = rag_manager.get_database_stats()
                    if stats.get("status") == "Active":
                        return f"✅ Using existing database | Documents: {stats['document_count']} | Tip: Only initialize if you have new documents", f"Documents: {stats['document_count']}"
            except:
                pass  # Continue with recreation if existing DB has issues

        # Step 1: Load documents
        success, doc_message, doc_count = load_knowledge_base(knowledge_base_path, file_pattern)

        if not success:
            return f"❌ {doc_message}", "Documents: 0"

        # Step 2: Create/Update vector database (recreate embeddings)
        db_success, db_message = create_vector_db()

        if db_success:
            return f"✅ Successfully initialized RAG database: {doc_count} documents loaded and indexed", f"Documents: {doc_count}"
        else:
            return f"⚠️ Documents loaded but database creation failed: {db_message}", f"Documents: {doc_count}"

    except Exception as e:
        return f"❌ Error initializing RAG database: {str(e)}", "Documents: 0"


def prepare_messages_with_rag(system, prompt):
    """
    Prepare messages for API call with RAG context included.

    Args:
        system (str): System message
        prompt (str): User prompt

    Returns:
        tuple: (messages, retrieved_context)
    """
    retrieved_docs = rag_manager.get_retrieval_context(prompt)  # Uses config.DEFAULT_RETRIEVAL_K
    if retrieved_docs:
        # Format retrieved context with similarity scores
        context_parts = []
        llm_context_parts = []

        for i, doc in enumerate(retrieved_docs):
            doc_type = doc.metadata.get('doc_type', 'unknown')
            similarity_score = doc.metadata.get('similarity_score', 'N/A')
            similarity_percentage = doc.metadata.get('similarity_percentage', 'N/A')

            # Format for user display (with scores)
            if isinstance(similarity_score, float):
                score_display = f"Distance: {similarity_score:.4f} | Similarity: {similarity_percentage:.1f}%"
            else:
                score_display = "Score: N/A"

            context_parts.append(
                f"**Document {i+1}** (Type: {doc_type}) | {score_display}\n"
                f"Content: {doc.page_content}\n"
                f"{'='*80}"
            )

            # Format for LLM context (without scores to reduce token usage)
            llm_context_parts.append(f"Document {i+1} (Type: {doc_type}):\n{doc.page_content}")

        # User-facing context with scores
        user_context = "\n\n".join(context_parts)

        # LLM context without scores
        llm_context = "\n\n".join(llm_context_parts)

        # Modify system message to include context for LLM
        enhanced_system = f"{system}\n\nRelevant context from knowledge base:\n{llm_context}\n\nPlease use this context to inform your response when relevant."

        return [
            {"role": "system", "content": enhanced_system},
            {"role": "user", "content": prompt}
        ], user_context
    else:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ], "No relevant documents found in knowledge base."

def prepare_messages(system, prompt):
    """
    Prepare standard messages for API call without RAG.

    Args:
        system (str): System message
        prompt (str): User prompt

    Returns:
        tuple: (messages, empty_context)
    """
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ], ""

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

    # RAG Section
    gr.Markdown("## RAG (Retrieval Augmented Generation)")
    with gr.Row():
        enable_rag = gr.Checkbox(value=config.DEFAULT_RAG_ENABLED, label="Enable RAG")
        rag_status = gr.Textbox(label="RAG Status", value="Not initialized", interactive=False)

    with gr.Accordion("RAG Configuration", open=False):
        with gr.Row():
            knowledge_base_path = gr.Textbox(
                label="Knowledge Base Path",
                value=config.DEFAULT_KNOWLEDGE_BASE_PATH,
                placeholder="Path to your knowledge base folder",
                info="Folder containing documents to index"
            )
            file_pattern = gr.Dropdown(
                choices=config.RAG_FILE_PATTERNS,
                value=config.RAG_FILE_PATTERNS[0],
                label="File Pattern",
                info="Type of files to process"
            )

        with gr.Row():
            initialize_rag_btn = gr.Button("🚀 Initialize RAG Database (Only needed for new/changed documents)", variant="primary")

        # Check if database exists on page load
        initial_status = "📂 No database found - click Initialize to create one"
        initial_docs = "0"
        if os.path.exists(config.DEFAULT_VECTOR_DB_NAME):
            initial_status = "📁 Database folder found - enable RAG to auto-load"

        with gr.Row():
            rag_status = gr.Textbox(label="RAG Status", value=initial_status, interactive=False)
            doc_count = gr.Textbox(label="Documents", value=initial_docs, interactive=False)

    with gr.Accordion("Vector Visualization", open=False):
        with gr.Row():
            viz_2d_btn = gr.Button("Generate 2D Plot")
            viz_3d_btn = gr.Button("Generate 3D Plot")

        with gr.Row():
            viz_2d_plot = gr.Plot(label="2D Vector Visualization")
            viz_3d_plot = gr.Plot(label="3D Vector Visualization")

    # Generate button to trigger API call
    generate_btn = gr.Button("Generate")

    # Output Section
    gr.Markdown("## Output Section")
    # Textbox for displaying reasoning tokens when available
    reasoning_output = gr.Textbox(label="Model Reasoning", lines=5, visible=True)
    # Textbox for displaying retrieved context when RAG is enabled
    context_output = gr.Textbox(label="Retrieved Context", lines=8, visible=False)
    # Textbox for displaying successful model responses
    response_output = gr.Textbox(label="Model Response", lines=10)
    # Textbox for displaying errors (hidden by default)
    error_output = gr.Textbox(label="Errors", visible=False)

    # Metadata Section
    gr.Markdown("## Metadata Section")
    # Markdown display for response metadata (model, time, tokens)
    metadata = gr.Markdown(label="Response Metadata")

    # Messages Accordion Section
    with gr.Accordion("Full Message Array", open=False):
        messages_output = gr.JSON(label="Conversation Messages", value=[])

    # Function for generating responses with error handling
    def generate_response(provider, model_dd, model_tb, base_url, api_key, system, prompt, temp, max_tok, top_p_val, freq_pen, pres_pen, stream, use_rag, custom_api_key):
        """
        Generate response from LLM API based on selected provider and parameters.
        Includes comprehensive error handling for API calls and RAG functionality.

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
            use_rag (bool): Whether to use RAG
            custom_api_key (str): Custom API key for RAG

        Yields:
            tuple: (gr.update for reasoning, gr.update for context, gr.update for response, gr.update for error, metadata_str, api_messages)
        """
        # Initialize variables
        full_response = ""
        reasoning = ""
        usage = None
        error_message = ""
        retrieved_context = ""

        # Initialize RAG if enabled
        if use_rag:
            rag_success, rag_message = initialize_rag_with_api_key(provider, api_key, custom_api_key)
            if not rag_success:
                error_message = f"RAG initialization failed: {rag_message}"

        # Determine model, API base URL, and API key using helper functions
        model = get_model(provider, model_dd, model_tb)
        api_base_url = get_api_base_url(provider, base_url)
        api_key_final = get_api_key(provider, api_key)

        # Create OpenAI client instance
        client = create_client(api_base_url, api_key_final)

        # Prepare messages for the API call (with or without RAG)
        if use_rag and not error_message:
            messages, retrieved_context = prepare_messages_with_rag(system, prompt)
        else:
            messages, retrieved_context = prepare_messages(system, prompt)

        # Record start time for response time calculation
        start_time = time.time()

        try:
            # Make API call using unified response handler
            last_usage = None
            last_reasoning = ""
            full_response = ""

            for response_content, response_usage, response_reasoning in handle_api_response(client, model, messages, temp, max_tok, top_p_val, freq_pen, pres_pen, stream):
                full_response = response_content
                if response_usage:
                    last_usage = response_usage
                if response_reasoning:
                    last_reasoning = response_reasoning

                # For streaming, yield each partial result
                if stream:
                    reasoning_value = last_reasoning if last_reasoning else "No reasoning tokens included"
                    yield (gr.update(value=reasoning_value),
                           gr.update(value=retrieved_context, visible=use_rag),
                           gr.update(value=full_response),
                           gr.update(value="", visible=False),
                           "",
                           messages)

            # Set final values for downstream processing
            usage = last_usage
            reasoning = last_reasoning

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

        # Format metadata for display
        metadata = format_metadata(model, provider, response_time, usage, error_message)

        # Yield with appropriate visibility for error output
        if error_message:
            yield (gr.update(value="No reasoning tokens included"),
                   gr.update(value="", visible=False),
                   gr.update(value=""),
                   gr.update(value=error_message, visible=True),
                   metadata,
                   messages)
        else:
            reasoning_value = reasoning if reasoning else "No reasoning tokens included"
            yield (gr.update(value=reasoning_value),
                   gr.update(value=retrieved_context, visible=use_rag),
                   gr.update(value=full_response),
                   gr.update(value="", visible=False),
                   metadata,
                   messages)

    # RAG Event Handlers
    initialize_rag_btn.click(
        fn=initialize_or_update_rag_database,
        inputs=[knowledge_base_path, file_pattern],
        outputs=[rag_status, doc_count]
    )

    viz_2d_btn.click(
        fn=create_2d_visualization,
        outputs=[viz_2d_plot]
    )

    viz_3d_btn.click(
        fn=create_3d_visualization,
        outputs=[viz_3d_plot]
    )

    # Update RAG status when RAG is enabled/disabled
    enable_rag.change(
        fn=auto_load_rag_database,
        inputs=[enable_rag, provider_radio],
        outputs=[rag_status]
    )

    # Event handler for the generate button
    # Calls generate_response function with all input values and updates output components
    generate_btn.click(
        fn=generate_response,
        inputs=[provider_radio, model_dropdown, model_textbox, custom_base_url, custom_api_key, system_message, user_prompt, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, streaming, enable_rag, custom_api_key],
        outputs=[reasoning_output, context_output, response_output, error_output, metadata, messages_output]
    )

# Main execution block
# Launches the Gradio application when the script is run directly
if __name__ == "__main__":
    demo.launch()