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
- **RAG (Retrieval Augmented Generation)**: Built-in support for knowledge-based question answering using vector databases.
- **Custom Embeddings**: Support for both OpenAI embeddings and custom embedding models (e.g., Ollama).
- **Vector Visualization**: 3D visualization of document embeddings for knowledge base exploration.
- **Flexible Document Processing**: Support for multiple file types and categorized knowledge organization.
- **MCP (Model Context Protocol)**: Integrated support for connecting to MCP servers, enabling LLMs to interact with external tools and data sources.
- **Manual Tool Testing**: Interactive interface for testing MCP tools with custom parameters before enabling automatic tool calling.
- **Automatic Tool Calling**: Allow LLMs to automatically discover and call tools from connected MCP servers during conversations.

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

## RAG (Retrieval Augmented Generation)

The application includes built-in RAG functionality that allows you to create a knowledge base and ask questions based on your documents.

### Setting Up Your Knowledge Base

1. **Create the knowledge base folder**: Create a `knowledge-base` folder in the project root (or specify a custom path in the UI).

2. **Organize your documents**: You can organize documents in subfolders for categorization:
   ```
   knowledge-base/
   ├── company-docs/
   │   ├── policies.md
   │   └── procedures.md
   ├── technical-docs/
   │   ├── api-guide.md
   │   └── troubleshooting.md
   └── faq.md
   ```

3. **Supported file types**:
   - **Recommended**: `.md` (Markdown) files for testing and general use
   - Also supported: `.txt` and `.pdf` files
   - Use the file pattern dropdown to select which types to process

### Using RAG

1. **Enable RAG**: Check the "Enable RAG" checkbox in the interface.

2. **Configure embeddings** (choose one):
   - **OpenAI**: Uses OpenAI's `text-embedding-3-small` model (1536 dimensions)
   - **Custom**: Use local embedding models like Ollama's `nomic-embed-text` (768 dimensions)

3. **For Custom embeddings** (e.g., Ollama):
   - Set Embedding Provider to "Custom"
   - Set Base URL to your local endpoint (e.g., `http://localhost:11434/v1`)
   - Set Model to your embedding model (e.g., `nomic-embed-text`)
   - Provide an API key (for Ollama, any value like "ollama" works)

4. **Initialize the database**:
   - Set the knowledge base path (default: `knowledge-base`)
   - Choose file pattern (default: `**/*.md`)
   - Click "🚀 Initialize RAG Database"
   - Wait for processing to complete

5. **Ask questions**: Once initialized, your questions will be enhanced with relevant context from your knowledge base.

### RAG Features

- **Document Processing**: Automatically chunks documents for optimal retrieval
- **Vector Database**: Uses Chroma for efficient similarity search
- **Context Display**: View retrieved documents in the expandable "RAG Information" section
- **3D Visualization**: Explore your knowledge base visually with the 3D vector plot
- **Embedding Flexibility**: Switch between OpenAI and custom embedding models
- **Status Monitoring**: Real-time feedback on database status, document count, and embedding dimensions

### Tips for Best Results

- **Use .md files** for testing as they're well-structured and easy to read
- **Organize by category** using subfolders for better document type tracking
- **Keep documents focused** - shorter, topic-specific documents often work better than very long ones
- **Consistent embedding models** - stick with one embedding model per knowledge base to avoid dimension mismatches

## MCP (Model Context Protocol)

The application includes comprehensive MCP (Model Context Protocol) support, allowing LLMs to interact with external tools and data sources through standardized servers.

### What is MCP?

MCP is an open protocol that standardizes how applications provide context to LLMs. It enables LLMs to securely access tools, databases, and services while maintaining a clear separation of concerns.

### Setting Up MCP Servers

1. **Create configuration file**: Create a `mcp-server.json` file in the project root:
   ```json
   {
     "mcpServers": {
       "filesystem": {
         "command": "npx",
         "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"]
       },
       "database": {
         "command": "npx",
         "args": ["-y", "@modelcontextprotocol/server-database", "postgresql://..."]
       }
     }
   }
   ```

2. **Supported server types**:
   - **Filesystem**: Access and manipulate files and directories
   - **Database**: Query and interact with databases
   - **Web**: Fetch and process web content
   - **Custom**: Any MCP-compliant server

### Using MCP Features

#### Manual Tool Testing

1. **Enable MCP**: Check the "Enable MCP" checkbox in the interface.

2. **Connect to servers**: Click "🔌 Connect to Servers" to establish connections to all servers defined in `mcp-server.json`.

3. **Test tools manually**:
   - Select a server from the dropdown
   - Choose a tool from the available tools
   - View the tool's parameters and schema
   - Enter parameters as JSON in the text box
   - Click "▶️ Execute Tool" to test the tool
   - View the results in the output section

#### Automatic Tool Calling by LLM

1. **Enable automatic tool calling**: After connecting to servers, check the "Enable Automatic Tool Calling by LLM" checkbox.

2. **Chat with enhanced capabilities**: The LLM can now:
   - Automatically discover available tools from all connected servers
   - Decide which tools to call based on your questions
   - Execute tools and use the results to generate responses
   - Make multiple tool calls in sequence if needed

3. **View tool call history**: All tool calls made during the conversation are displayed in the "MCP Tool Calls" section with:
   - Tool name and server
   - Parameters used
   - Results returned

### MCP Features

- **Multi-Server Support**: Connect to multiple MCP servers simultaneously
- **Tool Discovery**: Automatically discover and list available tools from each server
- **Schema Inspection**: View detailed parameter schemas for each tool
- **Interactive Testing**: Test tools manually before enabling automatic calling
- **Conversation Integration**: Seamlessly integrate tool calling into LLM conversations
- **Tool Call Tracking**: Monitor all tool calls made during conversations
- **Error Handling**: Graceful handling of connection failures and tool execution errors

### Example Use Cases

- **File Management**: Ask the LLM to read, write, or search files using the filesystem server
- **Data Analysis**: Query databases and analyze results with natural language
- **Web Research**: Fetch and process web content on demand
- **Custom Workflows**: Build custom MCP servers for domain-specific tasks

### MCP Server Resources

- Official MCP Servers: [https://github.com/modelcontextprotocol](https://github.com/modelcontextprotocol)
- Documentation: [https://modelcontextprotocol.io](https://modelcontextprotocol.io)
- Python SDK: [https://github.com/modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk)

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