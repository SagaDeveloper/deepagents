"""Middleware that enforces a hard cap on tavily_search tool calls.

RESEARCHER_INSTRUCTIONS already tells the research sub-agent to stop after 5
search tool calls, but that is advisory text only. This middleware makes the
limit deterministic by intercepting tavily_search calls once the budget has
been exhausted and returning a message telling the model to synthesize its
answer instead of calling the real tool.
"""

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

# Matches the "Hard Limits" section of RESEARCHER_INSTRUCTIONS in prompts.py:
# "Always stop: After 5 search tool calls if you cannot find the right sources."
MAX_SEARCH_CALLS_PER_SUBAGENT = 5


@wrap_tool_call
def enforce_search_budget(request, handler):
    """Block tavily_search calls once the sub-agent's search budget is spent.

    Counts prior tavily_search tool calls already present in the
    conversation's message history. If the count has reached
    MAX_SEARCH_CALLS_PER_SUBAGENT, short-circuits the call with a ToolMessage
    instructing the model to stop searching, instead of invoking the real
    tool via `handler`.
    """
    if request.tool_call["name"] != "tavily_search":
        return handler(request)

    prior_search_calls = 0
    for message in request.state.get("messages", []):
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            continue
        for tool_call in tool_calls:
            if tool_call.get("name") == "tavily_search":
                prior_search_calls += 1

    if prior_search_calls >= MAX_SEARCH_CALLS_PER_SUBAGENT:
        return ToolMessage(
            content=(
                "Search budget exhausted (5 calls). Stop searching and "
                "synthesize your answer now from what you already have."
            ),
            tool_call_id=request.tool_call["id"],
        )

    return handler(request)
