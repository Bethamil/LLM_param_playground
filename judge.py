"""
Judge Evaluation Module

This module provides LLM-based evaluation capabilities for assessing response quality.
It includes functions for creating judge clients, running evaluations, and formatting results.
"""

import json
from openai import OpenAI
import api_client


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
    api_base_url = api_client.get_api_base_url(judge_provider, judge_base_url)
    api_key_final = api_client.get_api_key(judge_provider, judge_api_key)
    return api_client.create_client(api_base_url, api_key_final)


def evaluate_with_judge(client, judge_model, messages, criteria, scale, temperature):
    """
    Evaluate a response using an LLM judge.

    Args:
        client (OpenAI): The judge client instance.
        judge_model (str): The judge model name.
        messages (list): Full conversation messages array including system, user, assistant, and tool messages.
        criteria (str): Evaluation criteria.
        scale (str): Scoring scale ("1-5", "1-10", "1-100").
        temperature (float): Judge temperature.

    Returns:
        tuple: (score, confidence, feedback, reasoning, usage)
    """
    max_score = int(scale.split('-')[1])

    # Format the full conversation for the judge
    conversation_json = json.dumps(messages, indent=2, ensure_ascii=False)

    # Truncate if extremely long (keep up to 20000 chars for judge context)
    if len(conversation_json) > 20000:
        conversation_json = conversation_json[:20000] + "\n... (truncated)"

    judge_prompt = f"""You are an expert judge evaluating an AI response. The full conversation (including system message, user query, tool calls, tool results, and final response) is provided below.

**Evaluation Criteria:** {criteria}

**Full Conversation:**
```json
{conversation_json}
```

Please evaluate the final assistant response based on the criteria above. Consider:
- Accuracy: Does the response accurately reflect information from tool results (if any)?
- Helpfulness: Is the response useful and complete?
- Clarity: Is the response well-structured and easy to understand?
- Relevance: Does the response address the user's query?

Please provide your evaluation in the following JSON format:
{{
    "score": <numerical score between 1 and {max_score}>,
    "confidence": <confidence in your score between 0 and 1>,
    "feedback": "<brief feedback on the response quality>",
    "reasoning": "<your reasoning process for the score>"
}}

Be strict but fair in your evaluation. If tool calls were made, verify that the response accurately represents the information retrieved from the tools."""

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
