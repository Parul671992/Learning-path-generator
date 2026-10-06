"""
Core pipeline for the Learning Path Generator (LangGraph version).

Extracted from learning_path_core.ipynb, cells 0-14, in the exact
same order as the notebook. Only the notebook-only display/testing
cells (diagram display, manual invoke/stream tests) were left out -
those still live in the notebook for your own experimentation.

No code was changed in this extraction - lines and order are identical
to the notebook. Inline comments tagged [ADDED], [FIX], [note], or
[FLAG, not fixed] mark points of historical interest (bugs found and
fixed earlier, deliberate design decisions, or things left as-is on
purpose) - they are comments only, not code changes.
"""

# ===== Cell 0 =====
# Warning control
import warnings
warnings.filterwarnings('ignore')

# ===== Cell 1 =====
from dotenv import load_dotenv
import os

_ = load_dotenv()

# ===== Cell 2 =====
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from typing import TypedDict, Literal, List
from langchain_openai import ChatOpenAI  # or ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import List
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
os.environ["HF_HUB_OFFLINE"] = "1"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ===== Cell 3 =====
class PlanItem(BaseModel):
    week: int = Field(description="Which week this item falls in")
    resource: str = Field(
        description="A single resource name or URL for this week. "
                     "If multiple resources apply, combine them into one "
                     "comma-separated string — do not return a list."
    )
    objective: str = Field(description="What the learner should achieve this week")

class WeeklyPlan(BaseModel):
    items: List[PlanItem]

# ===== Cell 4 =====
# [ADDED] Retry wrapper — structured-output LLM calls occasionally fail schema
# validation (e.g. model returns a list where a string was expected). This
# wrapper retries instead of crashing the whole run, with logging so repeated
# retries signal a prompt/schema that still needs tightening.
from typing import Callable, TypeVar

T = TypeVar("T")

def invoke_with_retry(structured_llm, messages, max_retries: int = 2, node_name: str = ""):
    """
    Calls structured_llm.invoke(messages) with retries on failure.
    Logs which attempt succeeded, so repeated retries signal a schema/prompt
    that may need tightening further.
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            result = structured_llm.invoke(messages)
            if attempt > 0:
                print(f"[{node_name}] Succeeded on retry attempt {attempt + 1}")
            return result
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                print(f"[{node_name}] Attempt {attempt + 1} failed ({type(e).__name__}), retrying...")
            continue
    print(f"[{node_name}] All {max_retries + 1} attempts failed.")
    raise last_error

# ===== Cell 5 =====
class LearningPathState(TypedDict):
    # User inputs
    topic: str
    time_weeks: int
    level: Literal["beginner", "intermediate", "expert"]
    goal: str  # "job hunting", "just learning", etc.
    # [ADDED] mode + docs_folder — the RAG extension. "web" uses the existing
    # Tavily search node; "documents" routes to a new retrieve node instead.
    # Both modes produce the same SearchResult shape, so everything downstream
    # (plan/generate/critique) needed zero changes to support this.
    mode: Literal["web", "documents"]
    docs_folder: str   # only used when mode == "documents" otherwise empty ""

    # Working state
    search_results: List[dict]      # raw resources found
    plan: List[PlanItem]
    generated_path: str              # final formatted output

    # Critique loop
    critique_feedback: str
    verdict: Literal["approve", "replan", "research"]
    iteration_count: int

# ===== Cell 6 =====
from langchain_community.tools.tavily_search import TavilySearchResults

search_tool = TavilySearchResults(max_results=5)

class SearchResult(BaseModel):
    title: str = Field(description="Title of the resource")
    url: str = Field(description="URL of the resource")
    summary: str = Field(description="Brief summary of what this resource covers")
    resource_type: str = Field(description="Type of resource, e.g. course, article, video, documentation")

class SearchResults(BaseModel):
    results: List[SearchResult]

def search_node(state: LearningPathState) -> LearningPathState:
    topic = state["topic"]
    level = state["level"]

    query = f"{level} level learning resources for {topic}"
    raw_results = search_tool.invoke({"query": query})

    structured_results = [
        SearchResult(
            title=r.get("title", "Untitled"),
            url=r.get("url", ""),
            summary=r.get("content", "")[:300],
            resource_type="unknown"  # Tavily doesn't classify this
        )
        for r in raw_results
    ]

    return {**state, "search_results": structured_results}

# ===== Cell 7 =====
def load_and_chunk_documents(folder_path: str):
    documents = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt") or filename.endswith(".md"):
            loader = TextLoader(filepath)
        else:
            continue
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)
    return chunks

# ===== Cell 8 =====
# [ADDED] retrieve_node — the RAG mode's resource-gathering node, parallel to
# search_node. Loads + chunks the user's documents, embeds them locally
# (free, no API key), and retrieves the chunks most relevant to the topic.
# Maps results into the same SearchResult schema search_node uses, tagged
# resource_type="user_document" so downstream nodes can tell the two apart
# if they ever need to, without requiring it.
def retrieve_node(state: LearningPathState) -> LearningPathState:
    topic = state["topic"]
    docs_folder = state.get("docs_folder", "./my_documents")

    chunks = load_and_chunk_documents(docs_folder)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    relevant_chunks = vectorstore.similarity_search(topic, k=5)

    structured_results = [
        SearchResult(
            title=f"{os.path.basename(chunk.metadata.get('source', 'unknown'))} "
                  f"(page {chunk.metadata.get('page', 'n/a')})",
            url=chunk.metadata.get("source", "unknown"),
            summary=chunk.page_content[:300],
            resource_type="user_document"
        )
        for chunk in relevant_chunks
    ]

    return {**state, "search_results": structured_results}

# ===== Cell 9 =====
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

structured_llm = llm.with_structured_output(WeeklyPlan)

def plan_node(state: LearningPathState) -> LearningPathState:
    topic = state["topic"]
    time_weeks = state["time_weeks"]
    level = state["level"]
    goal = state["goal"]
    search_results = state["search_results"]
    feedback = state.get("critique_feedback", "")

    resources_text = "\n".join(
    f"- {r.url}: {r.summary}"
    for r in search_results
        )

    system_prompt = (
        "You are an expert curriculum designer. Given a set of found resources, "
        "sequence them into a logical, prerequisite-aware learning path that fits "
        "the learner's time budget, level, and goal."
        # [FIX] Without this line, plan_node hallucinated generic, familiar
        # web courses instead of using the actually-retrieved resources —
        # this showed up specifically once RAG ("documents" mode) was added:
        # retrieval was correctly grounded in the user's files, but the plan
        # still invented unrelated web courses until this instruction was added.
        "IMPORTANT: Only use the resources explicitly provided below. Do not "
        "invent, assume, or add any other courses, articles, or platforms not "
        "listed in the provided resources."
    )

    user_prompt = (
        f"Topic: {topic}\n"
        f"Time available: {time_weeks} weeks\n"
        f"Level: {level}\n"
        f"Goal: {goal}\n"
        f"Available resources:\n{resources_text}\n"
    )
    if feedback:
        user_prompt += f"\nPrevious plan feedback to address: {feedback}"

    response: WeeklyPlan = invoke_with_retry(
        structured_llm,
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
        node_name="plan"
    )

    return {
        **state,
        "plan": response.items  # now a clean list of PlanItem objects
    }

# ===== Cell 10 =====
generate_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)

def generate_node(state: LearningPathState) -> LearningPathState:
    topic = state["topic"]
    level = state["level"]
    goal = state["goal"]
    time_weeks = state["time_weeks"]
    plan = state["plan"]  # List[PlanItem]

    plan_text = "\n".join(
        f"Week {item.week}: {item.resource} — {item.objective}"
        for item in plan
    )

    system_prompt = (
        "You are a friendly, encouraging learning coach. Turn the given weekly "
        "plan into a polished, motivating learning path write-up for the learner. "
        "Explain briefly why the sequence makes sense given their goal and level. "
        # [FIX] Originally said "use emojis sparingly" — the model took that as
        # license to use several. A soft qualifier produced inconsistent
        # behavior; this hard instruction fixed it completely. Same lesson as
        # the critique-approval-bar fix in Cell 11: vague instructions to an
        # LLM produce inconsistent behavior, explicit bounded ones don't.
        "Keep the tone professional and warm. Do not use emojis anywhere in the output."
    )


    user_prompt = (
        f"Topic: {topic}\n"
        f"Level: {level}\n"
        f"Goal: {goal}\n"
        f"Time: {time_weeks} weeks\n"
        f"Plan:\n{plan_text}\n"
    )

    response = generate_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    return {**state, "generated_path": response.content}

# ===== Cell 11 =====
class CritiqueResult(BaseModel):
    verdict: Literal["approve", "replan", "research"] = Field(
        description="approve if the plan is good, replan if sequencing/pacing is wrong "
                    "but resources are fine, research if resources themselves are poor/missing"
    )
    feedback: str = Field(
        description="Specific, actionable feedback explaining the verdict"
    )

structured_critique_llm = llm.with_structured_output(CritiqueResult)

def critique_node(state: LearningPathState) -> LearningPathState:
    topic = state["topic"]
    time_weeks = state["time_weeks"]
    level = state["level"]
    goal = state["goal"]
    plan = state["plan"]
    generated_path = state["generated_path"]
    iteration_count = state.get("iteration_count", 0)

    plan_text = "\n".join(
        f"Week {item.week}: {item.resource} — {item.objective}"
        for item in plan
    )

    # [FIX] This prompt originally said "be critical, not lenient" with no
    # defined bar for "good enough" — that caused critique to never approve
    # a plan, hitting MAX_ITERATIONS on every single run. The "IMPORTANT"
    # paragraph below (explicit approval criteria) fixed it: converged
    # naturally in 1-2 iterations afterward instead of always hitting the cap.
    system_prompt = (
        "You are a curriculum reviewer. Evaluate whether this learning path "
        "genuinely fits the learner's stated time budget, level, and goal. "
        "Check: Is pacing realistic? Are resources appropriate for the level? "
        "Does it actually serve the stated goal? "
        "IMPORTANT: This is a v1 learning plan, not a perfect one. Approve it if "
        "it is reasonably realistic and broadly appropriate, even if imperfect. "
        "Only recommend 'replan' for a genuine, significant mismatch (e.g., wildly "
        "unrealistic pacing, resources far above/below the learner's level, or a "
        "plan that doesn't serve the stated goal at all). Minor imperfections are "
        "expected and should not trigger a replan."
        )

    user_prompt = (
        f"Topic: {topic}\n"
        f"Time available: {time_weeks} weeks\n"
        f"Level: {level}\n"
        f"Goal: {goal}\n"
        f"Plan:\n{plan_text}\n"
        f"Generated write-up:\n{generated_path}\n"
    )

    result: CritiqueResult = invoke_with_retry(
        structured_critique_llm,
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
        node_name="critique"
    )

    return {
        **state,
        "verdict": result.verdict,
        "critique_feedback": result.feedback,
        "iteration_count": iteration_count + 1
    }

# ===== Cell 12 =====
MAX_ITERATIONS = 3

# [FIX] mode = state["mode"] + the ternary below were added after a real bug:
# this function originally hardcoded "research" -> "search" (web) regardless
# of mode, so a "documents" mode run that got a "research" verdict would
# silently switch to web search mid-run. Fixed by routing back to whichever
# node matches the original mode. A reminder that adding a second path
# through a graph means re-checking every existing conditional edge, not
# just adding the new node.
def route_after_critique(state: LearningPathState) -> str:
    verdict = state["verdict"]
    iteration_count = state["iteration_count"]
    mode = state["mode"]

    # Safety guard: force approval regardless of verdict once we hit the cap
    if iteration_count >= MAX_ITERATIONS:
        return "finalize"  # forced end — needs the disclaimer

    if verdict == "approve":
        return "finalize"  # clean end — finalize_node just passes it through
    elif verdict == "replan":
        return "plan"
    elif verdict == "research":
        return "search" if mode == "web" else "retrieve"

    return "finalize"

# ===== Cell 13 =====
def finalize_node(state: LearningPathState) -> LearningPathState:
    verdict = state["verdict"]
    iteration_count = state["iteration_count"]
    generated_path = state["generated_path"]

    if iteration_count >= MAX_ITERATIONS and verdict != "approve":
        # [FLAG, not fixed] This note uses a emoji (⚠️), which is inconsistent
        # with generate_node's "no emojis anywhere" rule (Cell 10) — left
        # exactly as in the original notebook per your instruction not to
        # change code, just flagging it in case you want it removed later.
        note = (
            "\n\n⚠️ Note: This plan went through the maximum number of "
            "refinement rounds and may still have some rough edges. "
            "Consider reviewing it manually."
        )
        generated_path = generated_path + note

    return {**state, "generated_path": generated_path}

# ===== Cell 14 =====
from langgraph.graph import StateGraph, END
# [note] StateGraph, END re-imported here — already imported in Cell 2.
# Harmless (Python just re-binds the same names), left exactly as in the
# original notebook.

# [ADDED] route_by_mode + set_conditional_entry_point below — the graph used
# to have a single fixed entry point (always "search"). This is what makes
# the RAG mode toggle actually take effect at the start of a run.
def route_by_mode(state: LearningPathState) -> str:
    return state["mode"]  # "web" or "documents"

# Initialize the graph with our state schema
builder = StateGraph(LearningPathState)

# Register all nodes
builder.add_node("search", search_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("plan", plan_node)
builder.add_node("generate", generate_node)
builder.add_node("critique", critique_node)
builder.add_node("finalize", finalize_node)

# Set the entry point
# [ADDED] conditional entry (was builder.set_entry_point("search") before RAG)
builder.set_conditional_entry_point(
    route_by_mode,
    {
        "web": "search",
        "documents": "retrieve"
    }
)

# Linear edges (unconditional)
builder.add_edge("search", "plan")
builder.add_edge("retrieve", "plan")  # [ADDED] retrieve -> plan edge, for RAG mode
builder.add_edge("plan", "generate")
builder.add_edge("generate", "critique")
builder.add_edge("finalize", END)

# Conditional edge after critique
builder.add_conditional_edges(
    "critique",
    route_after_critique,
    {
        "plan": "plan",
        "search": "search",
        "retrieve": "retrieve",  # [ADDED] retrieve as a valid route target, for RAG mode
        "finalize": "finalize"
    }
)

# Compile
conn = sqlite3.connect(":memory:", check_same_thread=False)
memory = SqliteSaver(conn)

graph = builder.compile(checkpointer=memory)
