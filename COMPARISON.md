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
