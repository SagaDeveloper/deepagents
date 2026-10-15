"""Orchestrator-level middleware for enforcing a per-trace tool-call budget.

This protects against traces that keep delegating further research (via the
`task()` tool) long after it has become clear the trace is trending toward
failure. Once the orchestrator has already made `MAX_ORCHESTRATOR_TOOL_CALLS`
tool calls of any kind, any further `task()` delegation is intercepted and the
model is redirected to finalize `/final_report.md` with whatever has already
been gathered, instead of continuing to spend on additional sub-agent rounds.
"""

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import AIMessage, ToolMessage

# Picked just above the median total tool-call count observed across recent
# harbor "research" category traces, so typical successful traces are not
# truncated while runaway traces are stopped before they accumulate excessive
# cost.
MAX_ORCHESTRATOR_TOOL_CALLS = 10


@wrap_tool_call
def enforce_trace_budget(request, handler):
    """Intercept further delegation once the trace's tool-call budget is spent.

    Counts all prior tool calls (of any name) already present in the
    orchestrator's own message history. Once that count has reached
    `MAX_ORCHESTRATOR_TOOL_CALLS`, any further `task()` call is short-circuited
    with a `ToolMessage` instructing the model to stop delegating and write
    `/final_report.md` using only what has already been gathered.
    """
    messages = request.state.get("messages", [])
    prior_tool_calls = sum(
        len(message.tool_calls)
        for message in messages
        if isinstance(message, AIMessage) and message.tool_calls
    )

    if prior_tool_calls >= MAX_ORCHESTRATOR_TOOL_CALLS and request.tool_call["name"] == "task":
        return ToolMessage(
            content=(
                "Trace tool budget exhausted. Stop delegating further "
                "research now and write /final_report.md using only what "
                "has already been gathered."
            ),
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
        )

    return handler(request)
