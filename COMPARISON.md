# LangGraph vs CrewAI: A Framework Comparison

This document is a deep dive into building the same Learning Path Generator twice — once in [LangGraph](https://github.com/langchain-ai/langgraph), once in [CrewAI](https://github.com/crewAIInc/crewAI) — to compare how each framework handles the same agentic workflow: search → plan → generate → critique, with a feedback loop that can send work back for revision.

## Architecture Comparison

| | LangGraph | CrewAI |
|---|---|---|
| **Model** | Explicit state graph with nodes and edges | Agents + Tasks, orchestrated by a Crew |
| **Data flow** | Shared state dict, read/written by any node | Explicit task-to-task `context=[...]` hand-offs |
| **Looping/routing** | Native — conditional edges route based on a function's return value | Not native in sequential mode — requires either hierarchical delegation (unreliable, see below) or a manual outer Python loop |
| **Control-flow enforcement** | Lives in code (Python function reads state, decides route) | Depends entirely on the approach — manual loop puts it in code too, but hierarchical mode leaves it to an LLM manager's judgment |
| **Structured output** | `.with_structured_output()` on the LLM — clean, provider-agnostic | `output_pydantic` on the Task — hit a real compatibility bug with Groq (see below) |
| **Persistence** | Built-in checkpointing (`SqliteSaver`) — can resume a run from any node | No native resume — a failed run must restart from task 1 |
| **Debugging experience** | `.stream()` shows state after every node | `verbose=True` shows a rich, colorized trace of every agent/task/tool step |

## What I Attempted: Hierarchical Mode

CrewAI's `Process.hierarchical` looked like the natural equivalent to LangGraph's conditional routing — a manager agent that could look at the critique verdict and decide whether to re-delegate work to the Researcher or Planner. In practice, this did not work reliably:

- The manager agent inherited tools from the first worker agent and frequently **executed the work itself** instead of delegating — a documented, unresolved CrewAI limitation ([crewAIInc/crewAI #2054](https://github.com/crewAIInc/crewAI/issues/2054), #4783, #2838). No amount of explicit backstory instruction ("you must always delegate, never execute directly") reliably prevented this.
- This burned through a disproportionate amount of the LLM token budget on manager reasoning and direct tool calls before the pipeline even reached the later tasks.

**Conclusion:** I pivoted to `Process.sequential` with a manual Python loop, mirroring the lesson learned earlier in the LangGraph build — that a hard stopping/routing rule needs to be enforced in code, not requested via prompt. CrewAI's hierarchical mode makes this concrete: routing decisions left to an LLM manager are a *request*, not a *guarantee*.

## Bugs Found (CrewAI + Groq)

Building the CrewAI version surfaced five distinct, genuine bugs — none of these were mistakes in the task/agent design; each required tracing an opaque error back to its root cause in a third-party library.

1. **`cache_breakpoint` field rejected by Groq.** CrewAI injects a `cache_breakpoint` field (meant for Anthropic's prompt caching) into every system message, but never strips it for non-Anthropic providers. Fixed with a targeted monkey-patch (`crewai.llms.cache.mark_cache_breakpoint`).

2. **Search tool schema too loose.** The built-in `SerperDevTool`'s schema didn't constrain the model tightly enough — Groq's `gpt-oss-120b` occasionally added extra parameters (`num`, `type`) that Groq's strict server-side validation then rejected. Fixed by writing a custom `BaseTool` with a single-field Pydantic schema, removing the model's ability to invent extra fields.

3. **Hierarchical manager tool inheritance** (see above) — a documented CrewAI limitation, not something fixable from the task/prompt side.

4. **`output_pydantic` internally broken with Groq.** CrewAI's structured-output mechanism calls an internal `"json"` tool to format the final answer, but this tool wasn't properly registered in the request sent to Groq, causing a tool-validation failure. Fixed by dropping `output_pydantic` entirely and instructing tasks to output raw JSON in the prompt, parsed manually with `json.loads()`.

5. **Stale task-output caching.** The most consequential bug: `Task` objects with static descriptions (no varying `{feedback}` placeholder) appeared to cache their LLM response, returning byte-for-byte identical output across loop iterations — even when the upstream `context` (the revised plan) had genuinely changed. This silently broke the critique loop: the Reviewer was correctly flagging the same never-updated write-up three times in a row, not being overly harsh. Fixed with `cache=False` on every task. This bug was diagnosed by noticing the Writer's output still referenced resource names ("Udacity Nanodegree") that no longer existed anywhere in the actual, updated plan JSON.

## A Recurring Theme: Vague Instructions Produce Inconsistent LLM Behavior

This showed up independently in three separate places across both frameworks, which makes it a genuinely generalizable lesson rather than an isolated fix:

- **LangGraph's critique node**, given "be critical, not lenient" with no defined bar for "good enough," never approved a plan — it hit the max-iteration cap on every run until the prompt was changed to state an explicit approval threshold.
- **LangGraph's generate node**, told to use emojis "sparingly," produced heavy emoji use; a hard "do not use emojis" instruction fixed it completely.
- **CrewAI's Researcher agent**, given an open-ended "find good resources" task, consistently used its full `max_iter` budget rather than stopping early — until given an explicit stopping condition ("if the first search returns 3-4 solid results, stop").

**Takeaway:** any point where an LLM makes a continue-vs-stop or approve-vs-reject decision needs an explicit, checkable criterion. Left implicit, the model defaults to either maximum caution (never approving) or maximum effort (never stopping) — both of which look like bugs but are really under-specified prompts.

## Cost & Reliability Observations

- CrewAI's agents, unconstrained, produced far more verbose output than requested (markdown tables, "how to use these" sections) — this directly caused repeated Groq rate-limit failures across a 4-task chained run. Tightening task descriptions to explicitly forbid this formatting fixed it.
- CrewAI's lack of resume support means every retry re-runs the entire pipeline from scratch — a failed run on task 4 of 4 still re-executes tasks 1-3. LangGraph's checkpointing would let a resumed run skip straight to the failed step.

## Verdict

Neither framework is strictly "better" — they suit different situations:

- **LangGraph** is the stronger choice when the workflow genuinely has cycles/branches that need reliable, code-enforced control flow — the graph model makes this explicit and guaranteed.
- **CrewAI** has a lower-friction mental model for role-based delegation (agents with personas, tasks with clear ownership) when the workflow is closer to a linear pipeline, but replicating LangGraph-style loops requires either an unreliable hierarchical mode or a manual Python loop that reimplements what LangGraph gives natively.
- Both frameworks, in this project, needed the same underlying lesson applied independently: **explicit, bounded instructions produce reliable agent behavior; vague ones don't** — regardless of which framework is doing the orchestration.

## MCP Server and Docker: Extraction, Local Client, and Containerized Deployment

The LangGraph version was further extended with an MCP server and a Docker build, as a hands-on exercise in the two layers that sit *around* an agent pipeline once it's built: how other tools call it (MCP), and where/how it actually runs (Docker).

### Extracting the pipeline out of the notebook

Both MCP and Docker need a plain, importable Python entrypoint — a notebook can't be imported by another program or run as a standalone process. `learning_path_core.py` holds the exact same code as the notebook (same lines, same order, extracted verbatim rather than rewritten, with inline comments flagging historical bug fixes without changing any code), confirmed to run correctly standalone before building anything on top of it.

### MCP: exposing the pipeline as a callable tool

`mcp_server.py` wraps the compiled graph in a single MCP tool, `generate_learning_path`, with `mode` exposed as a parameter so a caller can choose web search or document-grounded (RAG) generation. Connected to Claude Desktop using `stdio` transport (Claude Desktop launches the Python process locally and talks to it directly — no network, no hosting involved).

**Two real bugs found while setting this up:**

1. **A Windows-specific config location bug.** Claude Desktop was installed as an MSIX package, which redirects its actual config file to a sandboxed path (`AppData\Local\Packages\Claude_<id>\LocalCache\Roaming\Claude\claude_desktop_config.json`) rather than the conventional `%APPDATA%\Claude` location most documentation assumes. Editing the "expected" location silently had no effect — discovered by using the app's own "Edit config" button, which revealed the real path.

2. **An intermittent startup timeout, caused by a hidden network dependency.** The local embeddings model (`HuggingFaceEmbeddings`) checked in with the HuggingFace Hub on every server startup — even though the model was already downloaded and cached — and that check occasionally hung long enough on a slow network connection to exceed Claude Desktop's internal connection timeout, causing the server to appear "broken" with no code-level error to point to. Fixed with `HF_HUB_OFFLINE=1`, forcing it to use only the local cache.

### Docker: containerizing the same server for network deployment

The same `mcp_server.py` supports a second transport mode, `streamable-http`, controlled by one environment variable (`MCP_TRANSPORT`) that Claude Desktop's config never sets — so the local stdio setup is entirely unaffected by the Docker addition. In `streamable-http` mode the server listens on a real network port instead of talking over stdio to a local parent process, which is what makes it containerizable and reachable from outside the host machine at all.

**A real, iterative debugging sequence while getting the image to actually run (not just build):**

1. **Unconstrained PyTorch pulled several gigabytes of unused NVIDIA/CUDA packages.** `sentence-transformers` depends on PyTorch, and pip defaulted to the full GPU-enabled build. Fixed by pinning the CPU-only build explicitly (`--extra-index-url https://download.pytorch.org/whl/cpu`) — this container has no GPU to use, so the CUDA packages were pure waste, both in build time and image size.
2. **The `mcp` package resolved to a newer major version inside the container than the code was written against.** The local Anaconda environment had an older `mcp` 1.x installed from initial setup, which is why Claude Desktop worked fine locally — Docker, building fresh, pulled the newest available (`mcp` 2.x), which renamed `FastMCP` to `MCPServer` and broke the import. Fixed by pinning `mcp[cli]<2`. A clean illustration of why unpinned dependencies are risky: the bug wasn't introduced by Docker, it was *always* latent — Docker just doesn't share the same already-installed environment a local setup silently relies on.
3. **A genuinely unused import, previously flagged but left in place, caused a missing-dependency failure.** `learning_path_core.py` had a leftover `from langchain_openai import ChatOpenAI` from early exploration that was never actually called — harmless locally (already installed as a transitive dependency), but a hard failure in the trimmed `requirements-mcp.txt`, which deliberately didn't include it since nothing appeared to use it.
4. **A missing package for LangGraph's SQLite checkpointer.** `langgraph.checkpoint.sqlite` is published as a separate package, `langgraph-checkpoint-sqlite`, split out from the base `langgraph` package — not obvious from the import line alone.
5. **An API placement mismatch for `host`/`port`.** The installed `mcp` SDK version expects `host` and `port` to be passed to the `FastMCP` constructor, not to `.run()` — a `TypeError` revealed this directly, trivial once surfaced.

After these fixes, the container built and ran successfully (`Uvicorn running on http://0.0.0.0:8000`), and was confirmed reachable from outside the container with `curl http://localhost:8000/mcp`: a structured MCP protocol error response (rather than a connection failure) confirmed the server was correctly receiving and responding to requests over the network — `curl` isn't a full MCP client, so a protocol-level rejection is actually the expected, correct result for this kind of reachability check, not a failure.

### Why this matters for the framework/deployment comparison

This extension reinforced the same theme that ran through the LangGraph vs. CrewAI comparison: most of the real friction in building agentic systems isn't in the agent logic itself (both pipelines' core nodes worked essentially unchanged throughout) — it's in the surrounding infrastructure: environment assumptions that silently differ between "works on my machine" and a fresh environment, dependency version drift, and failure modes (a hanging network call, a sandboxed config path) that have nothing to do with LLMs or prompts at all. Docker didn't introduce these problems; it exposed ones that were already there, waiting in an unpinned requirements file and an environment nobody had explicitly specified.
