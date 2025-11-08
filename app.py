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
import time
from openai import AuthenticationError, RateLimitError, APIError, NotFoundError, APIConnectionError
from rag import RAGManager
from mcp_manager import MCPManager
import api_client
import judge
import ui_handlers

# Initialize RAG manager without specific embeddings (will be configured when needed)
rag_manager = RAGManager(db_name=config.DEFAULT_VECTOR_DB_NAME)

# Initialize MCP manager
mcp_manager = MCPManager(config_file=config.MCP_CONFIG_FILE)

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
        fn=ui_handlers.update_model_choices,
        inputs=[provider_radio, model_dropdown],
        outputs=[model_dropdown, model_textbox]
    )
    provider_radio.change(
        fn=ui_handlers.update_custom_fields,
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
                    success, _ = ui_handlers.load_existing_vector_db(rag_manager)
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

        # Max tool call iterations (only visible when automatic tool calling is enabled)
        max_tool_iterations = gr.Slider(
            minimum=1,
            maximum=20,
            value=config.DEFAULT_MAX_TOOL_ITERATIONS,
            step=1,
            label="Max Tool Call Iterations",
            info="Maximum number of tool call iterations allowed (default: 5)",
            visible=config.DEFAULT_MCP_TOOL_CALL_ENABLED
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
        fn=ui_handlers.update_model_choices,
        inputs=[judge_provider, judge_model_dropdown],
        outputs=[judge_model_dropdown, judge_model_textbox]
    )
    judge_provider.change(
        fn=ui_handlers.update_custom_fields,
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
        fn=ui_handlers.update_judge_visibility,
        inputs=enable_judge,
        outputs=[use_same_llm, judge_config_section, judge_column]
    )

    # Event handler for use same LLM checkbox
    use_same_llm.change(
        fn=ui_handlers.update_judge_llm_config_visibility,
        inputs=[enable_judge, use_same_llm],
        outputs=judge_llm_config
    )

    # Event handlers for MCP
    enable_mcp.change(
        fn=ui_handlers.update_mcp_visibility,
        inputs=enable_mcp,
        outputs=[mcp_config_section, mcp_output_column]
    )

    enable_mcp_tool_calling.change(
        fn=ui_handlers.update_max_tool_iterations_visibility,
        inputs=enable_mcp_tool_calling,
        outputs=[max_tool_iterations]
    )

    connect_mcp_btn.click(
        fn=lambda: ui_handlers.connect_to_mcp_servers(mcp_manager),
        outputs=[mcp_status, mcp_server_dropdown]
    )

    disconnect_mcp_btn.click(
        fn=lambda: ui_handlers.disconnect_from_mcp_servers(mcp_manager),
        outputs=[mcp_status, mcp_server_dropdown, mcp_tool_dropdown]
    )

    mcp_server_dropdown.change(
        fn=lambda server: ui_handlers.update_mcp_tools_dropdown(server, mcp_manager),
        inputs=mcp_server_dropdown,
        outputs=[mcp_tool_dropdown, mcp_tool_output]
    )

    mcp_tool_dropdown.change(
        fn=lambda server, tool: ui_handlers.get_tool_schema(server, tool, mcp_manager),
        inputs=[mcp_server_dropdown, mcp_tool_dropdown],
        outputs=mcp_tool_info
    )

    execute_tool_btn.click(
        fn=lambda server, tool, params: ui_handlers.execute_mcp_tool(server, tool, params, mcp_manager),
        inputs=[mcp_server_dropdown, mcp_tool_dropdown, mcp_params],
        outputs=mcp_tool_output
    )

    # Event handler for RAG enable/disable
    enable_rag.change(
        fn=lambda *args: ui_handlers.update_rag_visibility_and_status(*args, rag_manager),
        inputs=[enable_rag, provider_radio, embedding_provider, embedding_model, embedding_base_url, embedding_api_key],
        outputs=[rag_config_section, rag_column, rag_status]
    )

    # Event handler for embedding provider selection
    embedding_provider.change(
        fn=ui_handlers.update_embedding_custom_visibility,
        inputs=embedding_provider,
        outputs=embedding_custom_config
    )

    # Metadata Section (moved to bottom)
    gr.Markdown("## Metadata")
    # Markdown display for response metadata (model, time, tokens, judge info)
    metadata = gr.Markdown(label="Response Metadata")

    # Function for generating responses with error handling
    def generate_response(provider, model_dd, model_tb, base_url, api_key, system, prompt, temp, max_tok, top_p_val, freq_pen, pres_pen, stream, use_rag, custom_api_key,
                         enable_judge_flag, use_same_llm, judge_provider_val, judge_base_url_val, judge_api_key_val, judge_model_dd, judge_model_tb,
                         judge_temp, criteria, scale, embedding_provider, embedding_model, embedding_base_url, embedding_api_key,
                         enable_mcp_flag, enable_mcp_tool_calling, max_tool_iterations):
        """
        Generate response from LLM API based on selected provider and parameters.
        Includes comprehensive error handling for API calls, RAG functionality, and judge evaluation.
        """
        # Initialize variables
        full_response = ""
        reasoning = ""
        usage = None
        error_message = ""
        retrieved_context = ""

        # Initialize RAG if enabled
        if use_rag:
            rag_success, rag_message = ui_handlers.initialize_rag_with_api_key(
                provider, api_key, custom_api_key,
                embedding_provider, embedding_model, embedding_base_url, embedding_api_key, rag_manager
            )
            if not rag_success:
                error_message = f"RAG initialization failed: {rag_message}"
            else:
                # After successful embedding initialization, try to load existing database
                try:
                    import os
                    if os.path.exists(rag_manager.db_name):
                        load_success, load_message = rag_manager.load_existing_vector_database()
                        if not load_success:
                            print(f"Warning: Could not load existing database: {load_message}")
                except Exception as e:
                    print(f"Warning: Error loading existing database: {e}")

        # Judge evaluation variables
        judge_score_val = None
        judge_confidence_val = None
        judge_feedback_val = ""
        judge_reasoning_val = ""
        judge_data = None

        # Determine model, API base URL, and API key using helper functions
        model = api_client.get_model(provider, model_dd, model_tb)
        api_base_url = api_client.get_api_base_url(provider, base_url)
        api_key_final = api_client.get_api_key(provider, api_key)

        # Create OpenAI client instance
        client = api_client.create_client(api_base_url, api_key_final)

        # Prepare messages for the API call (with or without RAG)
        if use_rag and not error_message:
            messages, retrieved_context = api_client.prepare_messages_with_rag(system, prompt, rag_manager)
        else:
            messages = api_client.prepare_messages(system, prompt)
            retrieved_context = ""

        # Record start time for response time calculation
        start_time = time.time()

        # Prepare tools if MCP tool calling is enabled
        tools = None
        tool_calls_log = []
        if enable_mcp_flag and enable_mcp_tool_calling and mcp_manager.connected:
            tools = mcp_manager.format_tools_for_openai()

        try:
            # Make API call using unified response handler
            last_usage = None
            last_reasoning = ""
            full_response = ""
            conversation_messages_with_tools = None

            # Use unified handler for all cases (with or without tools, streaming or not)
            for response_content, response_usage, response_reasoning, tool_calls_made, conv_messages in api_client.handle_api_response(
                client, model, messages, temp, max_tok, top_p_val, freq_pen, pres_pen, tools, stream, max_tool_iterations, mcp_manager
            ):
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
                    if enable_judge_flag:
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
                           gr.update(visible=enable_judge_flag),
                           gr.update(value=tool_calls_log))

            # Set final values for downstream processing
            usage = last_usage
            reasoning = last_reasoning

            # For non-streaming, yield immediately with response and loading states for judge if enabled
            if not stream:
                reasoning_value = reasoning if reasoning else "No reasoning tokens included"
                response_time = time.time() - start_time
                metadata_temp = judge.format_metadata(model, provider, response_time, usage, "", judge_enabled=False)

                if enable_judge_flag:
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
                           gr.update(visible=enable_judge_flag),
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
            if enable_judge_flag:
                reasoning_value = reasoning if reasoning else "No reasoning tokens included"
                metadata_display = judge.format_metadata(model, provider, response_time, usage, error_message, judge_enabled=False)

                # Yield with loading states for judge fields
                yield (gr.update(value=reasoning_value),
                       gr.update(value=full_response),
                       gr.update(value="", visible=False),
                       metadata_display,
                       messages,
                       gr.update(value=retrieved_context),
                       gr.update(value=None),  # Number fields must remain None during loading
                       gr.update(value=None),
                       gr.update(value="🤖 Judge is evaluating the response..."),
                       gr.update(value="⏳ Processing evaluation criteria..."),
                       gr.update(visible=enable_judge_flag),
                       gr.update(value=tool_calls_log))

            # Perform judge evaluation if enabled
            if enable_judge_flag:
                judge_start_time = time.time()
                try:
                    # Use same LLM settings or judge-specific settings
                    if use_same_llm:
                        judge_model = model
                        judge_client = client
                        judge_provider_name = provider
                    else:
                        judge_model = api_client.get_model(judge_provider_val, judge_model_dd, judge_model_tb)
                        judge_client = judge.get_judge_client(judge_provider_val, judge_base_url_val, judge_api_key_val)
                        judge_provider_name = judge_provider_val

                    judge_score_val, judge_confidence_val, judge_feedback_val, judge_reasoning_val, judge_usage = judge.evaluate_with_judge(
                        judge_client, judge_model, messages, criteria, scale, judge_temp
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
                    judge_score_val = None
                    judge_confidence_val = None
                    judge_feedback_val = f"Judge evaluation failed: {str(e)}"
                    judge_reasoning_val = str(e)
                    judge_evaluation_time = time.time() - judge_start_time
                    judge_data = {
                        'model': 'N/A',
                        'provider': 'N/A',
                        'evaluation_time': judge_evaluation_time,
                        'same_llm': use_same_llm,
                        'usage': None
                    }

        # Format metadata for display
        if enable_judge_flag and judge_data is not None:
            metadata_display = judge.format_metadata(model, provider, response_time, usage, error_message, judge_enabled=True, judge_data=judge_data)
        else:
            metadata_display = judge.format_metadata(model, provider, response_time, usage, error_message, judge_enabled=False)

        # Yield with appropriate visibility for error output
        if error_message:
            yield (gr.update(value="No reasoning tokens included"),
                   gr.update(value=""),
                   gr.update(value=error_message, visible=True),
                   metadata_display,
                   messages,
                   gr.update(value=""),
                   gr.update(value=judge_score_val),
                   gr.update(value=judge_confidence_val),
                   gr.update(value=judge_feedback_val),
                   gr.update(value=judge_reasoning_val),
                   gr.update(visible=enable_judge_flag),
                   gr.update(value=tool_calls_log))
        else:
            reasoning_value = reasoning if reasoning else "No reasoning tokens included"
            yield (gr.update(value=reasoning_value),
                   gr.update(value=full_response),
                   gr.update(value="", visible=False),
                   metadata_display,
                   messages,
                   gr.update(value=retrieved_context),
                   gr.update(value=judge_score_val),
                   gr.update(value=judge_confidence_val),
                   gr.update(value=judge_feedback_val),
                   gr.update(value=judge_reasoning_val),
                   gr.update(visible=enable_judge_flag),
                   gr.update(value=tool_calls_log))

    # Preset Event Handlers
    def save_current_preset(name, *args):
        """Save current settings as a preset."""
        if not name or not name.strip():
            return "❌ Please enter a preset name!", gr.update(choices=ui_handlers.get_preset_names())

        settings = ui_handlers.collect_current_settings(*args)
        message = ui_handlers.save_preset(name.strip(), settings)
        return message, gr.update(choices=ui_handlers.get_preset_names())

    def load_selected_preset(name):
        """Load a selected preset."""
        if not name:
            return "❌ Please select a preset to load!", *ui_handlers.apply_preset_settings({})

        settings, message = ui_handlers.load_preset(name)
        if settings:
            return message, *ui_handlers.apply_preset_settings(settings)
        return message, *ui_handlers.apply_preset_settings({})

    def delete_selected_preset(name):
        """Delete a selected preset."""
        if not name:
            return "❌ Please select a preset to delete!", gr.update(choices=ui_handlers.get_preset_names())

        message = ui_handlers.delete_preset(name)
        return message, gr.update(choices=ui_handlers.get_preset_names())

    # Function to update preset dropdown
    def update_preset_dropdown():
        """Update the preset dropdown with current available presets."""
        return gr.update(choices=ui_handlers.get_preset_names())

    # Preset button event handlers
    save_preset_btn.click(
        fn=save_current_preset,
        inputs=[preset_name, provider_radio, model_dropdown, model_textbox, custom_base_url, custom_api_key,
                system_message, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, streaming,
                enable_rag, embedding_provider, embedding_model, embedding_base_url, embedding_api_key,
                knowledge_base_path, file_pattern, enable_judge, use_same_llm, judge_provider,
                judge_base_url, judge_api_key, judge_model_dropdown, judge_model_textbox,
                judge_temperature, judge_criteria, scoring_scale, enable_mcp, enable_mcp_tool_calling, max_tool_iterations],
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
                 judge_temperature, judge_criteria, scoring_scale, enable_mcp, enable_mcp_tool_calling, max_tool_iterations]
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
        fn=lambda *args: ui_handlers.initialize_or_update_rag_database(*args, rag_manager)[0],  # Only return status, ignore doc_count
        inputs=[knowledge_base_path, file_pattern, provider_radio, custom_api_key, embedding_provider, embedding_model, embedding_base_url, embedding_api_key],
        outputs=[rag_status]
    )

    viz_3d_btn.click(
        fn=lambda: ui_handlers.create_3d_visualization(rag_manager),
        outputs=[viz_3d_plot]
    )


    # Event handler for the generate button
    # Calls generate_response function with all input values and updates output components
    generate_btn.click(
        fn=generate_response,
        inputs=[provider_radio, model_dropdown, model_textbox, custom_base_url, custom_api_key, system_message, user_prompt, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, streaming, enable_rag, custom_api_key,
                enable_judge, use_same_llm, judge_provider, judge_base_url, judge_api_key, judge_model_dropdown, judge_model_textbox,
                judge_temperature, judge_criteria, scoring_scale, embedding_provider, embedding_model, embedding_base_url, embedding_api_key,
                enable_mcp, enable_mcp_tool_calling, max_tool_iterations],
        outputs=[reasoning_output, response_output, error_output, metadata, messages_output, context_output, judge_score, judge_confidence, judge_feedback, judge_reasoning, judge_column, mcp_tool_calls_output]
    )

# Main execution block
# Launches the Gradio application when the script is run directly
if __name__ == "__main__":
    demo.launch()
