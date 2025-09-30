"""
MCP Manager Module

This module provides a wrapper for MCP (Model Context Protocol) client functionality.
It manages connections to MCP servers, tool discovery, and tool execution.

Features:
- Connect to multiple MCP servers using stdio transport
- Discover available tools from all connected servers
- Execute tools with parameters
- Async context management for server connections
- Error handling and status reporting
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import config


class MCPManager:
    """
    Manager class for MCP (Model Context Protocol) client operations.

    Handles connection to multiple MCP servers, tool discovery, and execution.
    """

    def __init__(self, config_file: str = None):
        """
        Initialize the MCP Manager.

        Args:
            config_file (str): Path to MCP server configuration JSON file.
        """
        self.config_file = config_file or config.MCP_CONFIG_FILE
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack: Optional[AsyncExitStack] = None
        self.server_tools: Dict[str, List[Any]] = {}  # Maps server name to list of tools
        self.connected = False

    async def connect_to_servers(self) -> Tuple[bool, str]:
        """
        Connect to all MCP servers defined in the configuration file.

        Returns:
            Tuple[bool, str]: (success, message) indicating connection status
        """
        if not os.path.exists(self.config_file):
            return False, f"Configuration file not found: {self.config_file}"

        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON in configuration file: {str(e)}"
        except Exception as e:
            return False, f"Error reading configuration file: {str(e)}"

        if 'mcpServers' not in config_data:
            return False, "No 'mcpServers' key found in configuration"

        servers = config_data['mcpServers']
        if not servers:
            return False, "No servers defined in configuration"

        self.exit_stack = AsyncExitStack()
        connected_servers = []
        failed_servers = []

        for server_name, server_config in servers.items():
            try:
                # Extract command and args from server config
                command = server_config.get('command')
                args = server_config.get('args', [])
                env = server_config.get('env', None)

                if not command:
                    failed_servers.append(f"{server_name} (no command specified)")
                    continue

                # Setup server parameters
                server_params = StdioServerParameters(
                    command=command,
                    args=args,
                    env=env
                )

                # Establish stdio transport
                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                stdio, write = stdio_transport

                # Create session
                session = await self.exit_stack.enter_async_context(
                    ClientSession(stdio, write)
                )

                # Initialize session
                await session.initialize()

                # Store session
                self.sessions[server_name] = session

                # List available tools from this server
                try:
                    response = await session.list_tools()
                    self.server_tools[server_name] = response.tools
                    tool_count = len(response.tools)
                    connected_servers.append(f"{server_name} ({tool_count} tools)")
                except Exception as e:
                    self.server_tools[server_name] = []
                    connected_servers.append(f"{server_name} (0 tools, error: {str(e)})")

            except Exception as e:
                failed_servers.append(f"{server_name} ({str(e)})")

        if not connected_servers:
            self.connected = False
            return False, f"Failed to connect to any servers. Errors: {', '.join(failed_servers)}"

        self.connected = True
        status_msg = f"✅ Connected to {len(connected_servers)} server(s): {', '.join(connected_servers)}"

        if failed_servers:
            status_msg += f"\n⚠️ Failed to connect to: {', '.join(failed_servers)}"

        return True, status_msg

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers and cleanup resources."""
        if self.exit_stack:
            await self.exit_stack.aclose()
            self.exit_stack = None

        self.sessions.clear()
        self.server_tools.clear()
        self.connected = False

    def get_connected_servers(self) -> List[str]:
        """
        Get list of connected server names.

        Returns:
            List[str]: List of server names
        """
        return list(self.sessions.keys())

    def get_tools_for_server(self, server_name: str) -> List[Any]:
        """
        Get list of tools available from a specific server.

        Args:
            server_name (str): Name of the server

        Returns:
            List[Any]: List of tool objects
        """
        return self.server_tools.get(server_name, [])

    def get_all_tools(self) -> Dict[str, List[Any]]:
        """
        Get all tools from all connected servers.

        Returns:
            Dict[str, List[Any]]: Dictionary mapping server names to tool lists
        """
        return self.server_tools

    def get_tool_by_name(self, server_name: str, tool_name: str) -> Optional[Any]:
        """
        Get a specific tool by name from a server.

        Args:
            server_name (str): Name of the server
            tool_name (str): Name of the tool

        Returns:
            Optional[Any]: Tool object if found, None otherwise
        """
        tools = self.get_tools_for_server(server_name)
        for tool in tools:
            if tool.name == tool_name:
                return tool
        return None

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Call a tool on a specific server with given arguments.

        Args:
            server_name (str): Name of the server
            tool_name (str): Name of the tool to call
            arguments (Dict[str, Any]): Arguments to pass to the tool

        Returns:
            Tuple[bool, Any]: (success, result/error_message)
        """
        if server_name not in self.sessions:
            return False, f"Server '{server_name}' not connected"

        session = self.sessions[server_name]

        try:
            result = await session.call_tool(tool_name, arguments)
            return True, result
        except Exception as e:
            return False, f"Error calling tool: {str(e)}"

    def format_tools_for_openai(self) -> List[Dict[str, Any]]:
        """
        Format all tools as OpenAI function definitions for LLM tool calling.

        Returns:
            List[Dict[str, Any]]: List of tool definitions in OpenAI format
        """
        openai_tools = []

        for server_name, tools in self.server_tools.items():
            for tool in tools:
                # Create tool definition in OpenAI format
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": f"{server_name}__{tool.name}",  # Prefix with server name
                        "description": tool.description or f"Tool {tool.name} from {server_name}",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                }

                # Add input schema if available
                if hasattr(tool, 'inputSchema') and tool.inputSchema:
                    schema = tool.inputSchema
                    if isinstance(schema, dict):
                        if 'properties' in schema:
                            tool_def["function"]["parameters"]["properties"] = schema['properties']
                        if 'required' in schema:
                            tool_def["function"]["parameters"]["required"] = schema['required']

                openai_tools.append(tool_def)

        return openai_tools

    async def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Execute a tool call from an LLM tool call response.
        Tool name should be in format: server_name__tool_name

        Args:
            tool_name (str): Full tool name (server__tool format)
            arguments (Dict[str, Any]): Tool arguments

        Returns:
            Tuple[bool, Any]: (success, result/error_message)
        """
        # Parse server name and tool name
        if '__' not in tool_name:
            return False, f"Invalid tool name format: {tool_name}. Expected 'server__tool'"

        server_name, actual_tool_name = tool_name.split('__', 1)

        return await self.call_tool(server_name, actual_tool_name, arguments)

    def get_status(self) -> str:
        """
        Get current connection status as a formatted string.

        Returns:
            str: Status message
        """
        if not self.connected:
            return "❌ Not connected to any servers"

        server_count = len(self.sessions)
        total_tools = sum(len(tools) for tools in self.server_tools.values())

        status_lines = [f"✅ Connected to {server_count} server(s)"]
        status_lines.append(f"🔧 Total tools available: {total_tools}")

        for server_name, tools in self.server_tools.items():
            status_lines.append(f"  • {server_name}: {len(tools)} tool(s)")

        return "\n".join(status_lines)


# Utility functions for synchronous access in Gradio
def run_async(coro):
    """
    Helper function to run async coroutines in sync context.

    Args:
        coro: Coroutine to run

    Returns:
        Result of the coroutine
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)
