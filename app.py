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
from rag import RAGManager
from mcp_manager import MCPManager, run_async

# Preset management
PRESETS_FILE = "presets.json"

def load_presets():
    """Load presets from JSON file."""
    if os.path.exists(PRESETS_FILE):
        try:
            with open(PRESETS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_presets(presets):
    """Save presets to JSON file."""
    with open(PRESETS_FILE, 'w') as f:
        json.dump(presets, f, indent=2)

def save_preset(name, settings):
    """Save current settings as a preset."""
    presets = load_presets()
    presets[name] = settings
    save_presets(presets)
    return f"✅ Preset '{name}' saved successfully!"

def load_preset(name):
    """Load a preset and return the settings."""
    presets = load_presets()
    if name in presets:
        return presets[name], f"✅ Preset '{name}' loaded successfully!"
    return None, f"❌ Preset '{name}' not found!"

def delete_preset(name):
    """Delete a preset."""
    presets = load_presets()
    if name in presets:
        del presets[name]
        save_presets(presets)
        return f"✅ Preset '{name}' deleted successfully!"
    return f"❌ Preset '{name}' not found!"

def get_preset_names():
    """Get list of available preset names."""
    presets = load_presets()
    return list(presets.keys())

def collect_current_settings(provider, model_dd, model_tb, base_url, api_key, system, temp, max_tok, top_p_val, freq_pen, pres_pen, stream,
                           enable_rag, embedding_provider, embedding_model, embedding_base_url, embedding_api_key,
                           knowledge_base_path, file_pattern, enable_judge, use_same_llm, judge_provider,
                           judge_base_url, judge_api_key, judge_model_dd, judge_model_tb, judge_temp, criteria, scale):
    """Collect all current UI settings into a dictionary."""
    return {
        "provider": provider,
        "model_dropdown": model_dd,
        "model_textbox": model_tb,
        "custom_base_url": base_url,
        "custom_api_key": api_key,
        "system_message": system,
        "temperature": temp,
        "max_tokens": max_tok,
        "top_p": top_p_val,
        "frequency_penalty": freq_pen,
        "presence_penalty": pres_pen,
        "streaming": stream,
        "enable_rag": enable_rag,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_base_url": embedding_base_url,
        "embedding_api_key": embedding_api_key,
        "knowledge_base_path": knowledge_base_path,
        "file_pattern": file_pattern,
        "enable_judge": enable_judge,
        "use_same_llm": use_same_llm,
        "judge_provider": judge_provider,
        "judge_base_url": judge_base_url,
        "judge_api_key": judge_api_key,
        "judge_model_dropdown": judge_model_dd,
        "judge_model_textbox": judge_model_tb,
        "judge_temperature": judge_temp,
        "judge_criteria": criteria,
        "scoring_scale": scale
    }

def apply_preset_settings(settings):
    """Apply preset settings to all UI components."""
    return (
        gr.update(value=settings.get("provider", "OpenAI")),
        gr.update(value=settings.get("model_dropdown", None)),
        gr.update(value=settings.get("model_textbox", "")),
        gr.update(value=settings.get("custom_base_url", "")),
        gr.update(value=settings.get("custom_api_key", "")),
        gr.update(value=settings.get("system_message", "")),
        gr.update(value=settings.get("temperature", config.DEFAULT_TEMPERATURE)),
        gr.update(value=settings.get("max_tokens", config.DEFAULT_MAX_TOKENS)),
        gr.update(value=settings.get("top_p", config.DEFAULT_TOP_P)),
        gr.update(value=settings.get("frequency_penalty", config.DEFAULT_FREQUENCY_PENALTY)),
        gr.update(value=settings.get("presence_penalty", config.DEFAULT_PRESENCE_PENALTY)),
        gr.update(value=settings.get("streaming", config.DEFAULT_STREAMING)),
        gr.update(value=settings.get("enable_rag", config.DEFAULT_RAG_ENABLED)),
        gr.update(value=settings.get("embedding_provider", config.DEFAULT_EMBEDDING_PROVIDER)),
        gr.update(value=settings.get("embedding_model", config.DEFAULT_EMBEDDING_MODEL)),
        gr.update(value=settings.get("embedding_base_url", config.DEFAULT_CUSTOM_EMBEDDING_BASE_URL)),
        gr.update(value=settings.get("embedding_api_key", "")),
        gr.update(value=settings.get("knowledge_base_path", config.DEFAULT_KNOWLEDGE_BASE_PATH)),
        gr.update(value=settings.get("file_pattern", config.RAG_FILE_PATTERNS[0])),
        gr.update(value=settings.get("enable_judge", False)),
        gr.update(value=settings.get("use_same_llm", True)),
        gr.update(value=settings.get("judge_provider", "OpenAI")),
        gr.update(value=settings.get("judge_base_url", "")),
        gr.update(value=settings.get("judge_api_key", "")),
        gr.update(value=settings.get("judge_model_dropdown", None)),
        gr.update(value=settings.get("judge_model_textbox", "")),
        gr.update(value=settings.get("judge_temperature", 0.1)),
        gr.update(value=settings.get("judge_criteria", "Evaluate the response for: accuracy, helpfulness, clarity, and relevance to the query.")),
        gr.update(value=settings.get("scoring_scale", "1-10"))
    )

# Initialize RAG manager without specific embeddings (will be configured when needed)
rag_manager = RAGManager(db_name=config.DEFAULT_VECTOR_DB_NAME)

# Initialize MCP manager
mcp_manager = MCPManager(config_file=config.MCP_CONFIG_FILE)

def update_model_choices(provider, current_model_value=None):
    """
    Update the model dropdown choices based on the selected provider.
    For Custom provider, hide dropdown and show text input.
    Uses dynamic model fetching with fallback to hardcoded lists.
    Preserves the current model value if it's valid for the new provider.
    """
    if provider == "OpenAI":
        models = config.get_models_for_provider("OpenAI")
        # Keep current value if it's in the new models list, otherwise use first model
        new_value = current_model_value if current_model_value in models else (models[0] if models else None)
        return gr.update(choices=models, value=new_value, visible=True), gr.update(visible=False)
    elif provider == "OpenRouter":
        models = config.get_models_for_provider("OpenRouter")
        # Keep current value if it's in the new models list, otherwise use first model
        new_value = current_model_value if current_model_value in models else (models[0] if models else None)
        return gr.update(choices=models, value=new_value, visible=True), gr.update(visible=False)
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

def update_rag_visibility(enable_rag):
    """
    Show RAG configuration and output sections only when RAG is enabled.
    """
    return (
        gr.update(visible=enable_rag),  # rag_config_section
        gr.update(visible=enable_rag)   # rag_column (output)
    )

def update_rag_visibility_and_status(enable_rag, provider, embedding_provider, embedding_model, embedding_base_url, embedding_api_key):
    """
    Update RAG visibility and auto-load database status when RAG is enabled/disabled.
    """
    # Get visibility updates
    rag_config_visible = gr.update(visible=enable_rag)
    rag_column_visible = gr.update(visible=enable_rag)

    # Get status update with current embedding settings
    status_message = auto_load_rag_database(enable_rag, provider, embedding_provider, embedding_model, embedding_base_url, embedding_api_key)

    return (
        rag_config_visible,  # rag_config_section
        rag_column_visible,  # rag_column
        status_message       # rag_status
    )

def update_embedding_custom_visibility(embedding_provider):
    """
    Show custom embedding configuration only when Custom embedding provider is selected.
    """
    return gr.update(visible=embedding_provider == "Custom")

def update_judge_llm_config_visibility(enable_judge, use_same_llm):
    """
    Show judge LLM configuration only when judge is enabled and not using same LLM.
    """
    show_config = enable_judge and not use_same_llm
    return gr.update(visible=show_config)

def update_mcp_visibility(enable_mcp):
    """
    Show MCP configuration and output sections only when MCP is enabled.
    """
    return (
        gr.update(visible=enable_mcp),  # mcp_config_section
        gr.update(visible=enable_mcp)   # mcp_output_column
    )

def connect_to_mcp_servers():
    """
    Connect to all MCP servers defined in config file.
    Returns status message and updates server dropdown.
    """
    success, message = run_async(mcp_manager.connect_to_servers())

    if success:
        servers = mcp_manager.get_connected_servers()
        return message, gr.update(choices=servers, value=servers[0] if servers else None)
    else:
        return message, gr.update(choices=[], value=None)

def disconnect_from_mcp_servers():
    """
    Disconnect from all MCP servers.
    """
    run_async(mcp_manager.disconnect_all())
    return "🔌 Disconnected from all servers", gr.update(choices=[], value=None), gr.update(choices=[], value=None)

def update_mcp_tools_dropdown(server_name):
    """
    Update tools dropdown based on selected server.
    """
    if not server_name:
        return gr.update(choices=[], value=None), ""

    tools = mcp_manager.get_tools_for_server(server_name)
    tool_names = [tool.name for tool in tools]

    if tool_names:
        return gr.update(choices=tool_names, value=tool_names[0]), ""
    else:
        return gr.update(choices=[], value=None), "No tools available"

def get_tool_schema(server_name, tool_name):
    """
    Get the schema/parameters for a selected tool.
    """
    if not server_name or not tool_name:
        return "Select a server and tool to see parameters"

    tool = mcp_manager.get_tool_by_name(server_name, tool_name)
    if not tool:
        return "Tool not found"

    schema_info = []
    schema_info.append(f"**Tool:** {tool.name}")
    schema_info.append(f"**Description:** {tool.description or 'No description'}")

    if hasattr(tool, 'inputSchema') and tool.inputSchema:
        schema = tool.inputSchema
        if isinstance(schema, dict):
            schema_info.append(f"\n**Parameters Schema:**")
            schema_info.append(f"```json\n{json.dumps(schema, indent=2)}\n```")

    return "\n".join(schema_info)

def execute_mcp_tool(server_name, tool_name, params_json):
    """
    Execute an MCP tool with given parameters.
    """
    if not server_name or not tool_name:
        return "❌ Please select a server and tool"

    # Parse parameters
    try:
        if params_json.strip():
            params = json.loads(params_json)
        else:
            params = {}
    except json.JSONDecodeError as e:
        return f"❌ Invalid JSON parameters: {str(e)}"

    # Execute tool
    success, result = run_async(mcp_manager.call_tool(server_name, tool_name, params))

    if success:
        # Format result
        if hasattr(result, 'content') and result.content:
            content_parts = []
            for item in result.content:
                if hasattr(item, 'text'):
                    content_parts.append(item.text)
                elif hasattr(item, 'data'):
                    content_parts.append(f"Data: {item.data}")
            result_text = "\n".join(content_parts) if content_parts else str(result)
        else:
            result_text = str(result)

        return f"✅ Tool executed successfully:\n\n{result_text}"
    else:
        return f"❌ Tool execution failed: {result}"

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

def handle_api_response(client, model, messages, temp, max_tok, top_p_val, freq_pen, pres_pen, tools=None, stream=False, max_iterations=5):
    """
    Unified API response handler with streaming and tool calling support.

    This function handles:
    - Streaming responses (yields progressive updates)
    - Non-streaming responses
    - Tool calls (executes and continues conversation)
    - Multiple tool call iterations

    Args:
        client (OpenAI): The OpenAI client instance.
        model (str): The model name.
        messages (list): List of messages.
        temp (float): Temperature parameter.
        max_tok (int): Max tokens.
        top_p_val (float): Top-p parameter.
        freq_pen (float): Frequency penalty.
        pres_pen (float): Presence penalty.
        tools (list): Optional list of tool definitions.
        stream (bool): Whether to use streaming mode.
        max_iterations (int): Maximum tool call iterations.

    Yields:
        tuple: (response_content, usage, reasoning, tool_calls_made, conversation_messages)
    """
    iteration = 0
    conversation_messages = messages.copy()
    all_tool_calls = []
    final_response = ""
    final_usage = None
    final_reasoning = ""

    while iteration < max_iterations:
        iteration += 1

        try:
            # Make API call - always respect the stream parameter
            response = client.chat.completions.create(
                model=model,
                messages=conversation_messages,
                temperature=temp,
                max_completion_tokens=max_tok,
                top_p=top_p_val,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                tools=tools if tools else None,
                stream=stream
            )
        except Exception as e:
            yield f"Error: {str(e)}", None, "", all_tool_calls, conversation_messages
            return

        # Process response based on streaming mode
        if stream:
            # Streaming: collect response progressively
            content = ""
            reasoning = ""
            usage = None
            tool_calls_data = []

            for chunk in response:
                if not chunk.choices or len(chunk.choices) == 0:
                    continue
                delta = chunk.choices[0].delta

                # Collect content and yield progressive updates
                if delta.content:
                    content += delta.content
                    yield content, usage, reasoning, all_tool_calls, conversation_messages

                # Collect reasoning
                if hasattr(delta, 'reasoning') and delta.reasoning:
                    reasoning += delta.reasoning
                    yield content, usage, reasoning, all_tool_calls, conversation_messages

                # Collect tool calls
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        # Ensure we have enough slots
                        while len(tool_calls_data) <= tc_delta.index:
                            tool_calls_data.append({
                                'id': None,
                                'type': 'function',
                                'function': {'name': '', 'arguments': ''}
                            })

                        # Update tool call data
                        if tc_delta.id:
                            tool_calls_data[tc_delta.index]['id'] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_calls_data[tc_delta.index]['function']['name'] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_data[tc_delta.index]['function']['arguments'] += tc_delta.function.arguments

                # Collect usage
                if chunk.usage:
                    usage = chunk.usage
                    yield content, usage, reasoning, all_tool_calls, conversation_messages

            # Create message object from streamed data
            message_content = content
            message_reasoning = reasoning
            message_usage = usage

            # Convert tool_calls_data to proper format
            message_tool_calls = []
            if tool_calls_data:
                for tc_data in tool_calls_data:
                    message_tool_calls.append({
                        'id': tc_data['id'],
                        'type': tc_data['type'],
                        'function': {
                            'name': tc_data['function']['name'],
                            'arguments': tc_data['function']['arguments']
                        }
                    })
        else:
            # Non-streaming: extract response
            choice = response.choices[0]
            message = choice.message
            message_content = message.content or ""
            message_reasoning = getattr(message, 'reasoning', None) or ""
            message_usage = response.usage

            # Extract tool calls
            message_tool_calls = []
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tc in message.tool_calls:
                    message_tool_calls.append({
                        'id': tc.id,
                        'type': tc.type,
                        'function': {
                            'name': tc.function.name,
                            'arguments': tc.function.arguments
                        }
                    })

        # Store final values
        final_response = message_content
        final_reasoning = message_reasoning
        final_usage = message_usage

        # Add assistant message to conversation
        conversation_messages.append({
            "role": "assistant",
            "content": message_content,
            "tool_calls": message_tool_calls if message_tool_calls else None
        })

        # Check if there are tool calls to execute
        if message_tool_calls and tools:
            # Execute each tool call
            for tool_call in message_tool_calls:
                tool_name = tool_call['function']['name']
                tool_args_str = tool_call['function']['arguments']

                try:
                    tool_args = json.loads(tool_args_str)
                except json.JSONDecodeError:
                    tool_args = {}

                # Execute the tool via MCP
                success, result = run_async(mcp_manager.execute_tool_call(tool_name, tool_args))

                # Format tool result
                if success:
                    if hasattr(result, 'content') and result.content:
                        content_parts = []
                        for item in result.content:
                            if hasattr(item, 'text'):
                                content_parts.append(item.text)
                            elif hasattr(item, 'data'):
                                content_parts.append(str(item.data))
                        tool_result = "\n".join(content_parts) if content_parts else str(result)
                    else:
                        tool_result = str(result)
                else:
                    tool_result = f"Error: {result}"

                # Add tool result to conversation
                conversation_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call['id'],
                    "content": tool_result
                })

                # Track tool call
                all_tool_calls.append({
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result": tool_result
                })

            # Yield state after tool execution (keep content visible)
            yield final_response, final_usage, final_reasoning, all_tool_calls, conversation_messages

            # Continue loop to get next response after tool calls
            continue
        else:
            # No tool calls - we're done
            yield final_response, final_usage, final_reasoning, all_tool_calls, conversation_messages
            return

    # Max iterations reached
    yield final_response, final_usage, final_reasoning, all_tool_calls, conversation_messages

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

# RAG-specific functions
def initialize_rag_with_api_key(provider, api_key, custom_api_key, embedding_provider=None, embedding_model=None, embedding_base_url=None, embedding_api_key=None):
    """Initialize RAG manager with appropriate API key and embedding configuration."""
    # Determine the API key for embeddings
    if embedding_provider == "Custom":
        # Custom provider requires API key
        if not embedding_api_key or not embedding_api_key.strip():
            return False, "Custom embedding provider requires an API key"
        embedding_key = embedding_api_key
    else:
        # For OpenAI embeddings, always use OpenAI API key regardless of main provider
        if embedding_provider == "OpenAI":
            embedding_key = os.getenv("OPENAI_API_KEY") or (api_key if provider == "OpenAI" else None)
            if not embedding_key:
                return False, "OpenAI API key required for OpenAI embeddings. Set OPENAI_API_KEY environment variable."
        else:
            # This shouldn't happen, but fallback
            embedding_key = None

    if embedding_key:
        # Determine the correct model based on provider
        if embedding_provider == "Custom":
            model_to_use = embedding_model or config.DEFAULT_CUSTOM_EMBEDDING_MODEL
        else:
            model_to_use = config.DEFAULT_EMBEDDING_MODEL  # Use OpenAI default for OpenAI provider

        # Update RAG manager with new embedding configuration
        success = rag_manager.set_embeddings(
            api_key=embedding_key,
            provider=embedding_provider or config.DEFAULT_EMBEDDING_PROVIDER,
            model=model_to_use,
            base_url=embedding_base_url
        )
        if success:
            return True, f"RAG initialized with {embedding_provider or 'OpenAI'} embeddings"
        else:
            return False, "Failed to initialize embeddings - check console for details"
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

def auto_load_rag_database(rag_enabled, provider, embedding_provider=None, embedding_model=None, embedding_base_url=None, embedding_api_key=None):
    """Auto-load existing database when RAG is enabled."""
    if not rag_enabled:
        return "RAG disabled"

    # Check if database exists (including timestamped ones)
    import glob
    db_pattern = f"{config.DEFAULT_VECTOR_DB_NAME}*"
    existing_dbs = glob.glob(db_pattern)

    if existing_dbs:
        # Use the most recent database
        most_recent_db = sorted(existing_dbs)[-1]
        rag_manager.db_name = most_recent_db

        # Try to initialize embeddings and load database for visualization
        try:
            # Initialize with current embedding settings
            rag_success, rag_message = initialize_rag_with_api_key(
                provider, None, None,  # No main API keys needed
                embedding_provider, embedding_model, embedding_base_url, embedding_api_key
            )

            if rag_success:
                # Load the existing database
                load_success, load_message = rag_manager.load_existing_vector_database()
                if load_success:
                    stats = rag_manager.get_database_stats()
                    if stats.get("status") == "Active":
                        doc_count = stats.get('document_count', 'Unknown')
                        embedding_dims = stats.get('embedding_dimensions', 'Unknown')
                        return f"✅ Database loaded\n📄 Documents: {doc_count}\n📐 Dimensions: {embedding_dims}"
                    else:
                        return f"⚠️ Database loaded but inactive: {stats.get('status', 'Unknown')}"
                else:
                    return f"📁 Database found but couldn't load: {load_message}"
            else:
                return f"📁 Database found: {os.path.basename(most_recent_db)} (embeddings not configured)"
        except Exception as e:
            return f"📁 Database found: {os.path.basename(most_recent_db)} (load error: {str(e)})"
    else:
        return "📂 No database found - click Initialize to create one"

def create_3d_visualization():
    """Create 3D vector visualization."""
    fig = rag_manager.create_vector_visualization(dimensions=3)
    return fig


def initialize_or_update_rag_database(knowledge_base_path, file_pattern, provider, custom_api_key, embedding_provider, embedding_model, embedding_base_url, embedding_api_key):
    """
    Combined function to load documents and create/update vector database in one step.
    Always recreates embeddings to ensure new documents are included.

    Args:
        knowledge_base_path (str): Path to knowledge base folder
        file_pattern (str): File pattern to match
        provider (str): Main LLM provider
        custom_api_key (str): Custom API key for main provider
        embedding_provider (str): Embedding provider ("OpenAI" or "Custom")
        embedding_model (str): Embedding model name
        embedding_base_url (str): Custom embedding base URL
        embedding_api_key (str): Embedding API key

    Returns:
        tuple: (status_message, doc_count_message)
    """
    try:
        # Note: Always recreate the database when Initialize button is clicked
        # This ensures new documents are included

        # Step 1: Initialize embeddings with the provided configuration
        print(f"Step 1: Initializing embeddings...")
        rag_success, rag_message = initialize_rag_with_api_key(
            provider, None, custom_api_key,  # Use provided API keys
            embedding_provider, embedding_model, embedding_base_url, embedding_api_key
        )
        if not rag_success:
            return f"❌ Failed to initialize embeddings: {rag_message}", "0"

        # Step 2: Load documents
        success, doc_message, doc_count = load_knowledge_base(knowledge_base_path, file_pattern)

        if not success:
            return f"❌ {doc_message}", "Documents: 0"

        # Step 3: Create/Update vector database (recreate embeddings)
        print(f"Step 3: Creating vector database...")
        db_success, db_message = create_vector_db()

        if db_success:
            # Get detailed stats after successful creation
            try:
                stats = rag_manager.get_database_stats()
                if stats and stats.get("status") == "Active":
                    detailed_message = f"✅ Successfully initialized RAG database!\n📄 Documents processed: {doc_count}\n💾 Database status: Active"
                    return detailed_message, f"{doc_count}"
                else:
                    return f"✅ RAG database created but status unclear: {doc_count} documents processed", f"{doc_count}"
            except:
                return f"✅ Successfully initialized RAG database: {doc_count} documents loaded and indexed", f"{doc_count}"
        else:
            return f"⚠️ Documents loaded but database creation failed: {db_message}", f"{doc_count}"

    except Exception as e:
        return f"❌ Error initializing RAG database: {str(e)}", "0"


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

def evaluate_with_judge(client, judge_model, system, prompt, response, criteria, scale, temperature, tool_calls=None):
    """
    Evaluate a response using an LLM judge.

    Args:
        client (OpenAI): The judge client instance.
        judge_model (str): The judge model name.
        system (str): System message (may include RAG context if RAG was used).
        prompt (str): Original user prompt.
        response (str): The response to evaluate.
        criteria (str): Evaluation criteria.
        scale (str): Scoring scale ("1-5", "1-10", "1-100").
        temperature (float): Judge temperature.
        tool_calls (list): Optional list of tool calls made during generation.

    Returns:
        tuple: (score, confidence, feedback, reasoning, usage)
    """
    max_score = int(scale.split('-')[1])

    # Build tool calls information if available
    tool_calls_info = ""
    if tool_calls and len(tool_calls) > 0:
        tool_calls_info = "\n\n**MCP Tool Calls Made:**\n"
        for i, tool_call in enumerate(tool_calls, 1):
            tool_name = tool_call.get("tool", "Unknown")
            tool_args = tool_call.get("arguments", {})
            tool_result = tool_call.get("result", "No result")
            # Truncate long results for readability
            if isinstance(tool_result, str) and len(tool_result) > 500:
                tool_result = tool_result[:500] + "... (truncated)"
            tool_calls_info += f"{i}. **Tool:** {tool_name}\n"
            tool_calls_info += f"   **Arguments:** {json.dumps(tool_args, indent=2)}\n"
            tool_calls_info += f"   **Result:** {tool_result}\n\n"

    judge_prompt = f"""You are an expert judge evaluating AI responses. Please evaluate the following response based on these criteria: {criteria}

Original Query: {prompt}
Original System: {system}{tool_calls_info}
Response to evaluate: {response}

Please provide your evaluation in the following JSON format:
{{
    "score": <numerical score between 1 and {max_score}>,
    "confidence": <confidence in your score between 0 and 1>,
    "feedback": "<brief feedback on the response quality>",
    "reasoning": "<your reasoning process for the score>"
}}

Be strict but fair in your evaluation. Consider accuracy, helpfulness, clarity, and relevance.{" If tool calls were made, consider whether the response properly integrated the tool results." if tool_calls_info else ""}"""

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
custom_css = """
.large-button {
    height: 80px !important;
    font-size: 20px !important;
    font-weight: bold !important;
}
"""

with gr.Blocks(title="LLM Interactive Client", css=custom_css) as demo:
    # Main title
    gr.Markdown("# LLM Interactive Client")

    # Presets Section (moved to top)
    with gr.Accordion("Configuration Presets", open=False):
        gr.Markdown("Save and load different configuration settings for switching between setups.")

        with gr.Row():
            preset_name = gr.Textbox(
                label="Preset Name",
                placeholder="Enter a name for your preset",
                info="Name for saving presets"
            )
            save_preset_btn = gr.Button("💾 Save Preset", variant="secondary")

        with gr.Row():
            preset_dropdown = gr.Dropdown(
                choices=[],
                label="Available Presets",
                info="Select a preset to load or delete"
            )
            load_preset_btn = gr.Button("📂 Load Preset", variant="secondary")
            delete_preset_btn = gr.Button("🗑️ Delete Preset", variant="secondary")

        preset_status = gr.Textbox(
            label="Preset Status",
            value="",
            interactive=False,
            lines=2,
            max_lines=2
        )

    # Provider Selection Section
    gr.Markdown("## Provider Selection")
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
            visible=True,
            allow_custom_value=True
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
        inputs=[provider_radio, model_dropdown],
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
    with gr.Accordion("Model Parameters", open=False):
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
    enable_rag = gr.Checkbox(value=config.DEFAULT_RAG_ENABLED, label="Enable RAG", info="Enable Retrieval Augmented Generation for context-aware responses")

    # RAG configuration (visible only when RAG is enabled)
    with gr.Column(visible=False) as rag_config_section:
        # Embedding Provider Configuration
        embedding_provider = gr.Radio(
            choices=config.EMBEDDING_PROVIDERS,
            value=config.DEFAULT_EMBEDDING_PROVIDER,
            label="Embedding Provider",
            info="Choose the provider for generating document embeddings"
        )

        # Custom Embedding Configuration (visible only when Custom is selected)
        with gr.Column(visible=False) as embedding_custom_config:
            gr.Markdown("##### Custom Embedding Configuration")
            with gr.Row():
                embedding_base_url = gr.Textbox(
                    label="Embedding Base URL",
                    value=config.DEFAULT_CUSTOM_EMBEDDING_BASE_URL,
                    placeholder="http://localhost:11434/v1",
                    info="OpenAI-compatible embedding endpoint URL"
                )
                embedding_model = gr.Textbox(
                    label="Embedding Model",
                    value=config.DEFAULT_CUSTOM_EMBEDDING_MODEL,
                    placeholder="nomic-embed-text",
                    info="Name of the embedding model for your custom endpoint"
                )
            embedding_api_key = gr.Textbox(
                label="Embedding API Key",
                type="password",
                placeholder="Required for custom embedding endpoints",
                info="API key for your custom embedding service (e.g., for Ollama use any value like 'ollama')"
            )

        # Document Configuration
        with gr.Accordion("Document Processing", open=False):
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
                initialize_rag_btn = gr.Button("🚀 Initialize RAG Database (Only needed for new/changed documents)")

            # Check if database exists on page load and get detailed info
            initial_status = "📂 No database found - click Initialize to create one"
            initial_docs = "0"

            # Check for any vector database (including timestamped ones)
            import glob
            db_pattern = f"{config.DEFAULT_VECTOR_DB_NAME}*"
            existing_dbs = glob.glob(db_pattern)

            if existing_dbs:
                try:
                    # Find the most recent database (highest timestamp)
                    most_recent_db = sorted(existing_dbs)[-1]

                    # Update the RAG manager to use the most recent database
                    rag_manager.db_name = most_recent_db

                    # Try to get stats without fully loading
                    success, _ = load_existing_vector_db()
                    if success:
                        stats = rag_manager.get_database_stats()
                        if stats and stats.get("status") == "Active":
                            doc_count = stats.get('document_count', 'Unknown')
                            vector_count = stats.get('vector_count', 'Unknown')
                            initial_status = f"📁 Database ready to load\n📄 Documents: {doc_count}\n🔍 Vectors: {vector_count}"
                            initial_docs = str(doc_count)
                        else:
                            initial_status = "📁 Database folder found - enable RAG to auto-load"
                    else:
                        initial_status = "📁 Database folder found - enable RAG to auto-load"
                except:
                    initial_status = "📁 Database folder found - enable RAG to auto-load"

            # RAG Status Display
            rag_status = gr.Textbox(label="RAG Status", value=initial_status, interactive=False, lines=4, max_lines=4)

        # Vector Visualization
        with gr.Accordion("Vector Visualization", open=False):
            viz_3d_btn = gr.Button("Generate 3D Plot")
            viz_3d_plot = gr.Plot(label="3D Vector Visualization")

    # MCP Configuration Section
    gr.Markdown("## MCP (Model Context Protocol)")
    enable_mcp = gr.Checkbox(value=config.DEFAULT_MCP_ENABLED, label="Enable MCP", info="Enable Model Context Protocol for tool calling")

    # MCP configuration (visible only when MCP is enabled)
    with gr.Column(visible=False) as mcp_config_section:
        with gr.Row():
            connect_mcp_btn = gr.Button("🔌 Connect to Servers", variant="primary")
            disconnect_mcp_btn = gr.Button("🔌 Disconnect", variant="secondary")

        mcp_status = gr.Textbox(
            label="Connection Status",
            value="Not connected",
            interactive=False,
            lines=3
        )

        # Manual Tool Testing Section
        with gr.Accordion("Manual Tool Testing", open=False):
            gr.Markdown("Test MCP tools manually by selecting a server, tool, and providing parameters.")

            with gr.Row():
                mcp_server_dropdown = gr.Dropdown(
                    label="Select Server",
                    choices=[],
                    interactive=True,
                    info="Choose an MCP server"
                )
                mcp_tool_dropdown = gr.Dropdown(
                    label="Select Tool",
                    choices=[],
                    interactive=True,
                    info="Choose a tool from the selected server"
                )

            mcp_tool_info = gr.Markdown(
                value="Select a server and tool to see parameters",
                label="Tool Information"
            )

            mcp_params = gr.Code(
                label="Tool Parameters (JSON)",
                value='{\n  "param1": "value1",\n  "param2": "value2"\n}',
                language="json",
                lines=8
            )

            execute_tool_btn = gr.Button("▶️ Execute Tool", variant="primary")

            mcp_tool_output = gr.Textbox(
                label="Tool Output",
                lines=10,
                interactive=False,
                info="Result from tool execution"
            )

        # Enable automatic tool calling by LLM
        enable_mcp_tool_calling = gr.Checkbox(
            value=config.DEFAULT_MCP_TOOL_CALL_ENABLED,
            label="Enable Automatic Tool Calling by LLM",
            info="Allow the LLM to automatically call MCP tools during conversation"
        )

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
                    visible=True,
                    allow_custom_value=True
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
        inputs=[judge_provider, judge_model_dropdown],
        outputs=[judge_model_dropdown, judge_model_textbox]
    )
    judge_provider.change(
        fn=update_custom_fields,
        inputs=judge_provider,
        outputs=[judge_custom_row]
    )

    # Generate button to trigger API call
    with gr.Row():
        generate_btn = gr.Button("Generate", variant="primary", size="lg", scale=2, elem_classes="large-button")

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

    # RAG Output Section
    with gr.Column(visible=False) as rag_column:
        with gr.Accordion("RAG Information", open=False):
            # Textbox for displaying retrieved context when RAG is enabled
            context_output = gr.Textbox(
                label="Retrieved Context",
                lines=15,
                max_lines=20,
                info="Documents retrieved from the knowledge base",
                show_copy_button=True,
                autoscroll=False
            )

    # MCP Tool Calls Output Section
    with gr.Column(visible=False) as mcp_output_column:
        with gr.Accordion("MCP Tool Calls", open=True):
            # Display tool calls made by the LLM
            mcp_tool_calls_output = gr.JSON(
                label="Tool Calls Made",
                value=[],
                show_label=True
            )

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

    # Event handlers for MCP
    enable_mcp.change(
        fn=update_mcp_visibility,
        inputs=enable_mcp,
        outputs=[mcp_config_section, mcp_output_column]
    )

    connect_mcp_btn.click(
        fn=connect_to_mcp_servers,
        outputs=[mcp_status, mcp_server_dropdown]
    )

    disconnect_mcp_btn.click(
        fn=disconnect_from_mcp_servers,
        outputs=[mcp_status, mcp_server_dropdown, mcp_tool_dropdown]
    )

    mcp_server_dropdown.change(
        fn=update_mcp_tools_dropdown,
        inputs=mcp_server_dropdown,
        outputs=[mcp_tool_dropdown, mcp_tool_output]
    )

    mcp_tool_dropdown.change(
        fn=get_tool_schema,
        inputs=[mcp_server_dropdown, mcp_tool_dropdown],
        outputs=mcp_tool_info
    )

    execute_tool_btn.click(
        fn=execute_mcp_tool,
        inputs=[mcp_server_dropdown, mcp_tool_dropdown, mcp_params],
        outputs=mcp_tool_output
    )

    # Event handler for RAG enable/disable
    enable_rag.change(
        fn=update_rag_visibility_and_status,
        inputs=[enable_rag, provider_radio, embedding_provider, embedding_model, embedding_base_url, embedding_api_key],
        outputs=[rag_config_section, rag_column, rag_status]
    )

    # Event handler for embedding provider selection
    embedding_provider.change(
        fn=update_embedding_custom_visibility,
        inputs=embedding_provider,
        outputs=embedding_custom_config
    )

    # Metadata Section (moved to bottom)
    gr.Markdown("## Metadata")
    # Markdown display for response metadata (model, time, tokens, judge info)
    metadata = gr.Markdown(label="Response Metadata")

    # Function for generating responses with error handling
    def generate_response(provider, model_dd, model_tb, base_url, api_key, system, prompt, temp, max_tok, top_p_val, freq_pen, pres_pen, stream, use_rag, custom_api_key,
                         enable_judge, use_same_llm, judge_provider, judge_base_url, judge_api_key, judge_model_dd, judge_model_tb,
                         judge_temp, criteria, scale, embedding_provider, embedding_model, embedding_base_url, embedding_api_key,
                         enable_mcp, enable_mcp_tool_calling):
        """
        Generate response from LLM API based on selected provider and parameters.
        Includes comprehensive error handling for API calls, RAG functionality, and judge evaluation.

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
            enable_judge (bool): Whether to enable judge evaluation
            use_same_llm (bool): Whether to use same LLM for judging
            judge_provider (str): Judge provider
            judge_base_url (str): Judge base URL
            judge_api_key (str): Judge API key
            judge_model_dd (str): Judge model from dropdown
            judge_model_tb (str): Judge model from textbox
            judge_temp (float): Judge temperature
            criteria (str): Judge evaluation criteria
            scale (str): Judge scoring scale

        Yields:
            tuple: (gr.update for reasoning, gr.update for context, gr.update for response, gr.update for error, metadata_str, api_messages, judge outputs)
        """
        # Initialize variables
        full_response = ""
        reasoning = ""
        usage = None
        error_message = ""
        retrieved_context = ""

        # Initialize RAG if enabled
        if use_rag:
            rag_success, rag_message = initialize_rag_with_api_key(
                provider, api_key, custom_api_key,
                embedding_provider, embedding_model, embedding_base_url, embedding_api_key
            )
            if not rag_success:
                error_message = f"RAG initialization failed: {rag_message}"
            else:
                # After successful embedding initialization, try to load existing database
                try:
                    if os.path.exists(rag_manager.db_name):
                        load_success, load_message = rag_manager.load_existing_vector_database()
                        if not load_success:
                            print(f"Warning: Could not load existing database: {load_message}")
                except Exception as e:
                    print(f"Warning: Error loading existing database: {e}")

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

        # Prepare messages for the API call (with or without RAG)
        if use_rag and not error_message:
            messages, retrieved_context = prepare_messages_with_rag(system, prompt)
        else:
            messages, retrieved_context = prepare_messages(system, prompt)

        # Record start time for response time calculation
        start_time = time.time()

        # Prepare tools if MCP tool calling is enabled
        tools = None
        tool_calls_log = []
        if enable_mcp and enable_mcp_tool_calling and mcp_manager.connected:
            tools = mcp_manager.format_tools_for_openai()

        try:
            # Make API call using unified response handler
            last_usage = None
            last_reasoning = ""
            full_response = ""
            conversation_messages_with_tools = None

            # Use unified handler for all cases (with or without tools, streaming or not)
            for response_content, response_usage, response_reasoning, tool_calls_made, conv_messages in handle_api_response(client, model, messages, temp, max_tok, top_p_val, freq_pen, pres_pen, tools, stream):
                full_response = response_content
                if response_usage:
                    last_usage = response_usage
                if response_reasoning:
                    last_reasoning = response_reasoning
                if tool_calls_made:
                    tool_calls_log = tool_calls_made
                if conv_messages:
                    conversation_messages_with_tools = conv_messages

                # For streaming, yield progressive updates
                if stream:
                    reasoning_value = last_reasoning if last_reasoning else "No reasoning tokens included"

                    # Show loading states for judge fields during streaming if judge is enabled
                    if enable_judge:
                        judge_score_value = None
                        judge_confidence_value = None
                        judge_feedback_value = "⌛ Judge will evaluate once response is complete..."
                        judge_reasoning_value = "📝 Waiting for main response..."
                    else:
                        judge_score_value = None
                        judge_confidence_value = None
                        judge_feedback_value = ""
                        judge_reasoning_value = ""

                    yield (gr.update(value=reasoning_value),
                           gr.update(value=full_response),
                           gr.update(value="", visible=False),
                           "",
                           messages,
                           gr.update(value=retrieved_context),
                           gr.update(value=judge_score_value),
                           gr.update(value=judge_confidence_value),
                           gr.update(value=judge_feedback_value),
                           gr.update(value=judge_reasoning_value),
                           gr.update(visible=enable_judge),
                           gr.update(value=tool_calls_log))

            # Set final values for downstream processing
            usage = last_usage
            reasoning = last_reasoning

            # For non-streaming, yield immediately with response and loading states for judge if enabled
            if not stream:
                reasoning_value = reasoning if reasoning else "No reasoning tokens included"
                response_time = time.time() - start_time
                metadata_temp = format_metadata(model, provider, response_time, usage, "", judge_enabled=False)

                if enable_judge:
                    yield (gr.update(value=reasoning_value),
                           gr.update(value=full_response),
                           gr.update(value="", visible=False),
                           metadata_temp,
                           messages + [{"role": "assistant", "content": full_response}],
                           gr.update(value=retrieved_context),
                           gr.update(value=None),  # Number fields must remain None during loading
                           gr.update(value=None),
                           gr.update(value="🤖 Judge is evaluating the response..."),
                           gr.update(value="⏳ Processing evaluation criteria..."),
                           gr.update(visible=enable_judge),
                           gr.update(value=tool_calls_log))

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
            # If tool calling was used, use the full conversation messages which includes tool calls
            if conversation_messages_with_tools:
                messages = conversation_messages_with_tools
            else:
                # Otherwise just append the simple assistant response
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
                       gr.update(value=retrieved_context),
                       gr.update(value=None),  # Number fields must remain None during loading
                       gr.update(value=None),
                       gr.update(value="🤖 Judge is evaluating the response..."),
                       gr.update(value="⏳ Processing evaluation criteria..."),
                       gr.update(visible=enable_judge),
                       gr.update(value=tool_calls_log))

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

                    # Extract the actual system message that was used (includes RAG context if RAG was enabled)
                    actual_system_message = system
                    if messages and len(messages) > 0 and messages[0].get("role") == "system":
                        actual_system_message = messages[0]["content"]

                    judge_score, judge_confidence, judge_feedback, judge_reasoning, judge_usage = evaluate_with_judge(
                        judge_client, judge_model, actual_system_message, prompt, full_response, criteria, scale, judge_temp, tool_calls_log
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
                   gr.update(value=""),
                   gr.update(value=judge_score),
                   gr.update(value=judge_confidence),
                   gr.update(value=judge_feedback),
                   gr.update(value=judge_reasoning),
                   gr.update(visible=enable_judge),
                   gr.update(value=tool_calls_log))
        else:
            reasoning_value = reasoning if reasoning else "No reasoning tokens included"
            yield (gr.update(value=reasoning_value),
                   gr.update(value=full_response),
                   gr.update(value="", visible=False),
                   metadata,
                   messages,
                   gr.update(value=retrieved_context),
                   gr.update(value=judge_score),
                   gr.update(value=judge_confidence),
                   gr.update(value=judge_feedback),
                   gr.update(value=judge_reasoning),
                   gr.update(visible=enable_judge),
                   gr.update(value=tool_calls_log))

    # Preset Event Handlers
    def save_current_preset(name, *args):
        """Save current settings as a preset."""
        if not name or not name.strip():
            return "❌ Please enter a preset name!", gr.update(choices=get_preset_names())

        settings = collect_current_settings(*args)
        message = save_preset(name.strip(), settings)
        return message, gr.update(choices=get_preset_names())

    def load_selected_preset(name):
        """Load a selected preset."""
        if not name:
            return "❌ Please select a preset to load!", *apply_preset_settings({})

        settings, message = load_preset(name)
        if settings:
            return message, *apply_preset_settings(settings)
        return message, *apply_preset_settings({})

    def delete_selected_preset(name):
        """Delete a selected preset."""
        if not name:
            return "❌ Please select a preset to delete!", gr.update(choices=get_preset_names())

        message = delete_preset(name)
        return message, gr.update(choices=get_preset_names())

    # Function to update preset dropdown
    def update_preset_dropdown():
        """Update the preset dropdown with current available presets."""
        return gr.update(choices=get_preset_names())

    # Preset button event handlers
    save_preset_btn.click(
        fn=save_current_preset,
        inputs=[preset_name, provider_radio, model_dropdown, model_textbox, custom_base_url, custom_api_key,
                system_message, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, streaming,
                enable_rag, embedding_provider, embedding_model, embedding_base_url, embedding_api_key,
                knowledge_base_path, file_pattern, enable_judge, use_same_llm, judge_provider,
                judge_base_url, judge_api_key, judge_model_dropdown, judge_model_textbox,
                judge_temperature, judge_criteria, scoring_scale],
        outputs=[preset_status, preset_dropdown]
    )

    load_preset_btn.click(
        fn=load_selected_preset,
        inputs=[preset_dropdown],
        outputs=[preset_status, provider_radio, model_dropdown, model_textbox, custom_base_url, custom_api_key,
                 system_message, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, streaming,
                 enable_rag, embedding_provider, embedding_model, embedding_base_url, embedding_api_key,
                 knowledge_base_path, file_pattern, enable_judge, use_same_llm, judge_provider,
                 judge_base_url, judge_api_key, judge_model_dropdown, judge_model_textbox,
                 judge_temperature, judge_criteria, scoring_scale]
    )

    delete_preset_btn.click(
        fn=delete_selected_preset,
        inputs=[preset_dropdown],
        outputs=[preset_status, preset_dropdown]
    )

    # Update preset dropdown on page load
    demo.load(
        fn=update_preset_dropdown,
        outputs=[preset_dropdown]
    )

    # RAG Event Handlers
    initialize_rag_btn.click(
        fn=lambda *args: initialize_or_update_rag_database(*args)[0],  # Only return status, ignore doc_count
        inputs=[knowledge_base_path, file_pattern, provider_radio, custom_api_key, embedding_provider, embedding_model, embedding_base_url, embedding_api_key],
        outputs=[rag_status]
    )

    viz_3d_btn.click(
        fn=create_3d_visualization,
        outputs=[viz_3d_plot]
    )


    # Event handler for the generate button
    # Calls generate_response function with all input values and updates output components
    generate_btn.click(
        fn=generate_response,
        inputs=[provider_radio, model_dropdown, model_textbox, custom_base_url, custom_api_key, system_message, user_prompt, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, streaming, enable_rag, custom_api_key,
                enable_judge, use_same_llm, judge_provider, judge_base_url, judge_api_key, judge_model_dropdown, judge_model_textbox,
                judge_temperature, judge_criteria, scoring_scale, embedding_provider, embedding_model, embedding_base_url, embedding_api_key,
                enable_mcp, enable_mcp_tool_calling],
        outputs=[reasoning_output, response_output, error_output, metadata, messages_output, context_output, judge_score, judge_confidence, judge_feedback, judge_reasoning, judge_column, mcp_tool_calls_output]
    )

# Main execution block
# Launches the Gradio application when the script is run directly
if __name__ == "__main__":
    demo.launch()
