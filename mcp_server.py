"""
MCP server for the Learning Path Generator.

Exposes the existing LangGraph pipeline (learning_path_core.py) as a single
MCP tool, generate_learning_path, callable by any MCP client (Claude Desktop,
another agent, etc.). The mode parameter lets the caller choose "web" search
or "documents" (RAG over a local folder) per call - same toggle the notebook
already supports, just exposed over the protocol instead of hardcoded in a
notebook cell.

Run directly for local stdio testing (e.g. from Claude Desktop):
    python mcp_server.py

Run as a network service (e.g. inside Docker), set MCP_TRANSPORT=streamable-http
first. Claude Desktop's config is untouched either way - it doesn't set this
variable, so it keeps using the stdio default exactly as before.
"""

import os
import uuid

from mcp.server.fastmcp import FastMCP

from learning_path_core import graph

# [FIX for Docker] host/port belong on the FastMCP constructor, not on
# .run() - passing them to .run() raised TypeError: FastMCP.run() got an
# unexpected keyword argument 'host' on this mcp SDK version. They're only
# meaningful for streamable-http mode; FastMCP ignores them for stdio.
_transport = os.environ.get("MCP_TRANSPORT", "stdio")
_host = os.environ.get("MCP_HOST", "0.0.0.0")
_port = int(os.environ.get("MCP_PORT", "8000"))

mcp = FastMCP("learning-path-generator", host=_host, port=_port)


@mcp.tool()
def generate_learning_path(
    topic: str,
    level: str,
    goal: str,
    time_weeks: int,
    mode: str = "web",
    docs_folder: str = "",
) -> str:
    """
    Generate a personalized, critiqued, week-by-week learning path for a topic.

    Args:
        topic: What the learner wants to study, e.g. "Machine Learning Engineering".
        level: One of "beginner", "intermediate", "expert".
        goal: The learner's goal, e.g. "job hunting", "just learning".
        time_weeks: How many weeks the plan should span.
        mode: "web" to search the live web for resources, or "documents" to
            retrieve from a local folder of the user's own files instead.
        docs_folder: Required when mode="documents" - path to a folder of
            .txt, .md, or .pdf files to ground the plan in. Ignored for
            mode="web".

    Returns:
        The final, critiqued learning path as a markdown-formatted string.
    """
    initial_state = {
        "topic": topic,
        "time_weeks": time_weeks,
        "level": level,
        "goal": goal,
        "mode": mode,
        "docs_folder": docs_folder,
        "search_results": [],
        "plan": [],
        "generated_path": "",
        "critique_feedback": "",
        "verdict": "approve",
        "iteration_count": 0,
    }

    # Unique thread_id per call so checkpointed state from one request
    # never bleeds into another.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    final_state = graph.invoke(initial_state, config=config)
    return final_state["generated_path"]


if __name__ == "__main__":
    # [ADDED for Docker] transport switch. Defaults to "stdio" - Claude
    # Desktop's config never sets MCP_TRANSPORT, so this is a no-op change
    # for the existing local setup. Docker's entrypoint sets MCP_TRANSPORT=
    # streamable-http instead, which makes this listen on a real network
    # port (host/port set above, on the FastMCP object itself) rather than
    # talking over stdio to a locally-launching parent process.
    if _transport == "streamable-http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run()
