"""
UI Handlers Module

This module contains all Gradio UI state management functions including:
- Configuration preset management (save/load/delete)
- UI component visibility updates
- MCP server and tool management UI
- RAG database initialization and visualization UI
"""

import gradio as gr
import json
import os
import glob
import config
from mcp_manager import MCPManager, run_async


# ============================================================================
# PRESET MANAGEMENT
# ============================================================================

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
                           judge_base_url, judge_api_key, judge_model_dd, judge_model_tb, judge_temp, criteria, scale,
                           enable_mcp, enable_mcp_tool_calling, max_tool_iterations):
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
        "scoring_scale": scale,
        "enable_mcp": enable_mcp,
        "enable_mcp_tool_calling": enable_mcp_tool_calling,
        "max_tool_iterations": max_tool_iterations
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
        gr.update(value=settings.get("scoring_scale", "1-10")),
        gr.update(value=settings.get("enable_mcp", config.DEFAULT_MCP_ENABLED)),
        gr.update(value=settings.get("enable_mcp_tool_calling", config.DEFAULT_MCP_TOOL_CALL_ENABLED)),
        gr.update(value=settings.get("max_tool_iterations", config.DEFAULT_MAX_TOOL_ITERATIONS))
    )


# ============================================================================
# UI VISIBILITY UPDATES
# ============================================================================

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


def update_max_tool_iterations_visibility(enable_mcp_tool_calling):
    """
    Show max tool iterations slider only when automatic tool calling is enabled.
    """
    return gr.update(visible=enable_mcp_tool_calling)


# ============================================================================
# MCP UI HANDLERS
# ============================================================================

def connect_to_mcp_servers(mcp_manager):
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


def disconnect_from_mcp_servers(mcp_manager):
    """
    Disconnect from all MCP servers.
    """
    run_async(mcp_manager.disconnect_all())
    return "🔌 Disconnected from all servers", gr.update(choices=[], value=None), gr.update(choices=[], value=None)


def update_mcp_tools_dropdown(server_name, mcp_manager):
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


def get_tool_schema(server_name, tool_name, mcp_manager):
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


def execute_mcp_tool(server_name, tool_name, params_json, mcp_manager):
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


# ============================================================================
# RAG UI HANDLERS
# ============================================================================

def initialize_rag_with_api_key(provider, api_key, custom_api_key, embedding_provider, embedding_model, embedding_base_url, embedding_api_key, rag_manager):
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


def load_knowledge_base(knowledge_base_path, file_pattern, rag_manager):
    """Load documents from knowledge base."""
    if not knowledge_base_path:
        return False, "Please specify a knowledge base path", "No documents loaded"

    success, message, doc_count = rag_manager.load_documents_from_folder(knowledge_base_path, file_pattern)
    status_msg = f"Status: {message}"
    return success, status_msg, f"Documents: {doc_count}" if success else "Documents: 0"


def create_vector_db(rag_manager):
    """Create vector database from loaded documents."""
    success, message = rag_manager.create_vector_database()
    return success, f"Vector DB: {message}"


def load_existing_vector_db(rag_manager):
    """Load existing vector database."""
    success, message = rag_manager.load_existing_vector_database()
    return success, f"Vector DB: {message}"


def auto_load_rag_database(rag_enabled, provider, embedding_provider, embedding_model, embedding_base_url, embedding_api_key, rag_manager):
    """Auto-load existing database when RAG is enabled."""
    if not rag_enabled:
        return "RAG disabled"

    # Check if database exists (including timestamped ones)
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
                embedding_provider, embedding_model, embedding_base_url, embedding_api_key, rag_manager
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


def update_rag_visibility_and_status(enable_rag, provider, embedding_provider, embedding_model, embedding_base_url, embedding_api_key, rag_manager):
    """
    Update RAG visibility and auto-load database status when RAG is enabled/disabled.
    """
    # Get visibility updates
    rag_config_visible = gr.update(visible=enable_rag)
    rag_column_visible = gr.update(visible=enable_rag)

    # Get status update with current embedding settings
    status_message = auto_load_rag_database(enable_rag, provider, embedding_provider, embedding_model, embedding_base_url, embedding_api_key, rag_manager)

    return (
        rag_config_visible,  # rag_config_section
        rag_column_visible,  # rag_column
        status_message       # rag_status
    )


def initialize_or_update_rag_database(knowledge_base_path, file_pattern, provider, custom_api_key, embedding_provider, embedding_model, embedding_base_url, embedding_api_key, rag_manager):
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
        rag_manager: RAG manager instance

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
            embedding_provider, embedding_model, embedding_base_url, embedding_api_key, rag_manager
        )
        if not rag_success:
            return f"❌ Failed to initialize embeddings: {rag_message}", "0"

        # Step 2: Load documents
        success, doc_message, doc_count = load_knowledge_base(knowledge_base_path, file_pattern, rag_manager)

        if not success:
            return f"❌ {doc_message}", "Documents: 0"

        # Step 3: Create/Update vector database (recreate embeddings)
        print(f"Step 3: Creating vector database...")
        db_success, db_message = create_vector_db(rag_manager)

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


def create_3d_visualization(rag_manager):
    """Create 3D vector visualization."""
    fig = rag_manager.create_vector_visualization(dimensions=3)
    return fig
