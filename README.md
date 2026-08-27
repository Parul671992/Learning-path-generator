# Learning Path Generator

An AI agent that generates a personalized, week-by-week learning path for any topic — and iteratively critiques and refines its own output before presenting it to the user. Built with [LangGraph](https://github.com/langchain-ai/langgraph) to explore stateful, cyclical agent orchestration (as opposed to a simple one-shot LLM call).

A CrewAI implementation of the same system is planned as a comparison exercise — see [Framework Comparison](#framework-comparison-langgraph-vs-crewai) below.

## What it does

Given a **topic**, a **time budget** (in weeks), a **level** (beginner / intermediate / expert), and a **goal** (e.g. job hunting, general learning), the system:

1. **Searches** the web for real, current learning resources on the topic
2. **Plans** a week-by-week sequence of those resources, respecting the time budget and level
3. **Generates** a polished, human-readable write-up of the plan
4. **Critiques** its own output against the original constraints, and either:
   - **Approves** it and finalizes, or
   - **Sends it back to re-plan** (if pacing/sequencing is the problem), or
   - **Sends it back to re-search** (if the resources themselves are inadequate)

This loop repeats (capped at a max iteration count) until the plan is genuinely good, or the cap is hit — in which case the final output is still returned, with a disclaimer noting it may need manual review.

## Architecture

```
START → search → plan → generate → critique → (conditional routing)
              ↑           ↑                        ├─ approve → finalize → END
              └───────────┴────── replan/research ──┤
                                                      └─ max iterations → finalize → END
```

Built as a `StateGraph` with 5 nodes (`search`, `plan`, `generate`, `critique`, `finalize`) and one conditional edge that routes based on the critique agent's verdict.

## Setup

**1. Clone the repo and install dependencies**
```bash
git clone <your-repo-url>
cd learning-path-generator
pip install -r requirements.txt
```

**2. Get a free Groq API key**
Sign up at [console.groq.com](https://console.groq.com) — no credit card required.

**3. Create a `.env` file in the project root**
```
GROQ_API_KEY=your_key_here
```
(`.env` is gitignored — never commit real API keys.)

**4. Run the notebook**
Open `learning_path_generator.ipynb` in Jupyter and run all cells top to bottom.

## Design decisions & lessons learned

A few deliberate choices worth calling out — and what I learned building this:

- **Temperature varies by node role.** `plan` and `critique` use `temperature=0` (deterministic, evaluative tasks); `generate` uses `temperature=0.7` (creative, user-facing writing). One LLM config for every node would have been simpler but worse — matching temperature to the node's job produces more reliable structured output where it matters and more natural prose where it doesn't.

- **Structured output (Pydantic schemas) for anything a downstream node consumes; free text for anything a human reads.** `search` and `plan` both use `.with_structured_output()` so their results are type-safe and machine-parseable. `generate`'s output is deliberately free text, since it's the final human-facing artifact.

- **Critique calibration was the hardest part of this project, and the most instructive.** My first version of the critique prompt said "be critical, not lenient" with no defined bar for "good enough" — this caused the critique agent to *never* approve a plan, hitting the max-iteration cap on every run. The fix was adding explicit approval criteria ("approve unless there's a genuine, significant mismatch — minor imperfections are expected"). This is a real, generalizable lesson: **vague qualifiers given to an LLM produce inconsistent behavior; explicit, bounded instructions produce reliable behavior.** The same fix pattern applied to `generate`'s tone (a soft "use emojis sparingly" instruction was replaced with a hard "do not use emojis" instruction, which was followed consistently). Iteration counts across debugging: 5/5 (never converged) → 3/3 (still capping) → 2/3 (genuine replan then approve, converges naturally).

- **Search results replace rather than append on a re-search loop.** When critique routes back to `search` with feedback about poor resource quality, the new search results replace the old ones entirely rather than merging. This is a simplifying v1 assumption — a v2 could compare old vs. new and keep the better ones.

- **A retry wrapper around structured-output calls.** LLM outputs are probabilistic — even with a tight schema, a call can occasionally fail validation (e.g., the model returning a list where a string was expected). Rather than fixing this at the prompt level alone, I added a small retry wrapper (`invoke_with_retry`) with logging, so occasional validation failures self-heal instead of crashing the whole run. This is standard practice for any system built on non-deterministic model output.

- **Checkpointing (LangGraph's `SqliteSaver`).** The graph is compiled with a checkpointer, so state is saved after every node execution, keyed by a `thread_id`. This isn't just plumbing — it's what would enable resuming a long-running plan generation, inspecting intermediate state, or building a human-in-the-loop review step later.

## Known limitations (v1)

- `resource_type` on search results is left as `"unknown"` — classifying it properly would need an extra LLM call per search, which I skipped for cost/latency reasons given search can run multiple times per session. See tradeoff note in code comments.
- Checkpointing currently uses in-memory SQLite (`:memory:`), so state doesn't persist across kernel restarts. Switching to a file-based DB is a one-line change if cross-session persistence is needed.
- Single LLM provider (Groq). No fallback if Groq's API is unavailable.
- No human-in-the-loop step yet — the critique loop is fully autonomous. A natural v2 addition, enabled by the checkpointing already in place.

## Tech stack

- **LangGraph** — stateful agent orchestration
- **Groq** (`openai/gpt-oss-120b`) — LLM inference, free tier
- **Tavily** — web search
- **Pydantic** — structured output schemas

## Framework Comparison: LangGraph vs CrewAI

*(In progress — a CrewAI implementation of this same system is being built to compare against this LangGraph version on: control flow expressiveness, debugging experience, boilerplate, conditional branching support, and cost/latency for equivalent functionality.)*
