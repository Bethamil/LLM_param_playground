"""
API Client Module

This module provides a clean interface for interacting with LLM APIs (OpenAI, OpenRouter, Custom).
It handles client creation, API calls, streaming responses, and tool calling orchestration.
"""

import os
import json
import config
from openai import OpenAI


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
    Prepare standard messages for API call without RAG.

    Args:
        system (str): System message
        prompt (str): User prompt

    Returns:
        list: List of message dictionaries
    """
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]


def prepare_messages_with_rag(system, prompt, rag_manager):
    """
    Prepare messages for API call with RAG context included.

    Args:
        system (str): System message
        prompt (str): User prompt
        rag_manager: RAG manager instance for retrieving context

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


def handle_api_response(client, model, messages, temp, max_tok, top_p_val, freq_pen, pres_pen, tools=None, stream=False, max_iterations=5, mcp_manager=None):
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
        mcp_manager: MCP manager instance for tool execution.

    Yields:
        tuple: (response_content, usage, reasoning, tool_calls_made, conversation_messages)
    """
    from mcp_manager import run_async

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
        if message_tool_calls and tools and mcp_manager:
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
