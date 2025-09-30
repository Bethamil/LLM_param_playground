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
- **mcp_manager.py**: MCP (Model Context Protocol) client manager for connecting to servers, discovering tools, and executing tool calls
- **rag.py**: RAG (Retrieval Augmented Generation) manager for vector database operations and document processing
- **requirements.txt**: Python dependencies (gradio, openai, python-dotenv, mcp, langchain, chromadb, etc.)

### Key Architecture Patterns

**Provider System**: The application uses a modular provider system where each LLM provider (OpenAI, OpenRouter, Custom) has its own configuration and model fetching logic in `config.py`.

**Dynamic Model Fetching**: Models are fetched dynamically from provider APIs using cached requests with fallback to hardcoded lists if API calls fail. The caching system (`_cached_models`) prevents repeated API calls.

**OpenAI Client Pattern**: All providers use the OpenAI Python library as a unified interface, even for non-OpenAI providers like OpenRouter, by configuring custom base URLs.

**Streaming Support**: The application supports both streaming and non-streaming responses for real-time output display.

**RAG System**: Built-in Retrieval Augmented Generation using Chroma vector database for knowledge-based question answering with support for multiple embedding providers.

**MCP Integration**: Full Model Context Protocol support allowing LLMs to interact with external tools and data sources through standardized MCP servers with both manual testing and automatic tool calling capabilities.

### Configuration Management
- Provider settings and model lists are centralized in `config.py`
- API keys are handled through environment variables for security
- Default parameters are defined as constants (temperature, max tokens, etc.)
- Model caching prevents repeated API calls during a session
- MCP server configuration stored in `mcp-server.json` with support for multiple servers
- RAG knowledge base path and embedding configurations are customizable

### Error Handling
The application includes comprehensive error handling for:
- Authentication errors (invalid API keys)
- Rate limiting
- Network connectivity issues
- API errors and provider-specific issues
- Invalid model selections
- MCP server connection failures and tool execution errors
- RAG vector database initialization and query errors

### UI Architecture
- Uses Gradio Blocks for responsive layout
- Conditional visibility for custom provider fields, RAG configuration, MCP configuration
- Real-time parameter adjustment with sliders
- Progressive response display for streaming
- Metadata display for response time and token usage
- Accordion sections for RAG context, MCP tool calls, and judge evaluation
- Interactive MCP tool testing interface with server/tool selection and JSON parameter input

## MCP (Model Context Protocol) Integration

### MCP Manager (`mcp_manager.py`)
The `MCPManager` class handles all MCP-related operations:
- **Server Connection**: Connects to MCP servers via stdio transport using configuration from `mcp-server.json`
- **Tool Discovery**: Automatically discovers available tools from each connected server
- **Tool Execution**: Executes tools with parameters and returns results
- **OpenAI Format Conversion**: Converts MCP tools to OpenAI function calling format for seamless LLM integration
- **Async Management**: Uses async/await pattern with proper context management

### MCP Configuration
Create `mcp-server.json` in the project root:
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"]
    }
  }
}
```

### MCP UI Flow
1. User enables MCP checkbox → Shows MCP configuration section
2. User clicks "Connect to Servers" → `MCPManager.connect_to_servers()` reads config and establishes connections
3. User selects server → Tool dropdown populates with available tools
4. User selects tool → Schema/parameters displayed
5. User can manually test tool OR enable automatic tool calling
6. When automatic tool calling enabled, LLM receives tools in OpenAI format and can call them during conversation
7. Tool calls are tracked and displayed in the "MCP Tool Calls" output section

### Tool Calling Integration
The `handle_api_response_with_tools()` function manages the tool calling loop:
1. Makes API call with tools parameter
2. Checks if LLM wants to call any tools
3. Executes tools via `MCPManager.execute_tool_call()`
4. Adds tool results to conversation
5. Continues until LLM returns final response (max 5 iterations)
6. All tool calls are tracked and returned for display

## RAG (Retrieval Augmented Generation)

### RAG Manager (`rag.py`)
Handles vector database operations for knowledge-based QA using Chroma and LangChain.

### Key Features
- Document chunking and embedding
- Vector similarity search
- Support for multiple embedding providers (OpenAI, custom/Ollama)
- 3D visualization of embeddings
- Database statistics and monitoring