"""
graph.py — the LangGraph workflow.

MAIN GRAPH
    START
      -> planner        (LLM splits topic into 3-5 sub-questions)
      -> approve_plan   (interrupt: the human edits / approves the plan)
      -> fan_out        (one Send per sub-question)
      -> researcher xN  (run IN PARALLEL; each is the subgraph below)
      -> writer         (merges findings into a cited markdown report)
    -> END

RESEARCHER SUBGRAPH (self-correcting RAG, one copy per sub-question)
    START -> retrieve -> grade --(route_after_grade)--> answer -> END
                ^                       |
                |                       v
                +------------------- rewrite
    retrieve : search Chroma with the current query (no LLM)
    grade    : LLM picks which chunks actually help answer the question
    route    : relevant chunks found -> answer
               none found, rewrites left -> rewrite (then search again)
               none found, no rewrites left -> answer ("not covered")
    rewrite  : LLM writes a better search query
    answer   : LLM answers from the relevant chunks only, with citations

The main graph is compiled with an InMemorySaver checkpointer, which is what
lets interrupt() pause the run and Command(resume=...) continue it later on
the same thread_id. The subgraph shares that checkpointer automatically.
"""

import logging
import operator
import re
import threading
import time
from functools import lru_cache
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from pydantic import BaseModel, Field

import config
from retriever import search_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(threadName)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("research_graph")
# Silence chatty HTTP / library loggers so our own lines stand out.
for noisy in ("httpx", "sentence_transformers", "chromadb"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# ===========================================================================
# Helpers
# ===========================================================================
@lru_cache(maxsize=None)
def get_llm(role: str):
    """
    Build (once) the chat model for a role, with that role's temperature.
    Created lazily so the graph can be imported/compiled without an API key.
    """
    kwargs = {
        "temperature": config.TEMPERATURES[role],
        # Retry 429 rate-limit errors after the wait Groq asks for.
        "max_retries": config.MAX_RETRIES,
    }
    # reasoning_effort only exists on reasoning models (gpt-oss); other
    # models would reject it.
    if "gpt-oss" in config.MODEL:
        kwargs["reasoning_effort"] = config.REASONING_EFFORT
    return init_chat_model(config.MODEL, **kwargs)


# Citation tags look like [doc3.pdf, p.3] (built in retriever.py).
CITATION = re.compile(r"\[[^\[\]\n]+?, p\.\d+\]")


def extract_citations(text: str) -> list[str]:
    """All citation tags in `text`, in order of first appearance, no duplicates."""
    return list(dict.fromkeys(CITATION.findall(text)))


def strip_unknown_citations(text: str, allowed: set[str]) -> str:
    """
    Remove any citation tag that is not in `allowed` (i.e. not from a chunk
    the researcher actually retrieved). The hard guarantee behind
    "never invent a citation".
    """
    def keep_or_drop(match: re.Match) -> str:
        tag = match.group(0)
        if tag in allowed:
            return tag
        log.warning("Removed invented citation: %s", tag)
        return ""

    return CITATION.sub(keep_or_drop, text)


# ===========================================================================
# RESEARCHER SUBGRAPH (self-correcting RAG)
# ===========================================================================
class ResearcherInput(TypedDict):
    """What one researcher receives from fan_out's Send."""
    topic: str
    sub_question: str


class ResearcherOutput(TypedDict):
    """
    What one researcher hands back to the main graph: ONLY `findings`.
    Limiting the output matters: if every parallel researcher also returned
    `topic`, they would all write the same main-graph key at once -> error.
    """
    findings: Annotated[list, operator.add]


class ResearcherState(ResearcherInput, ResearcherOutput, total=False):
    """
    The researcher's private working memory (input + output + these).
    total=False: these keys don't exist yet when the researcher starts;
    the nodes below fill them in.
    """
    query: str                   # the search query for the current attempt
    rewrites: int                # how many times the query has been rewritten
    tried_queries: list[str]     # every query used so far (shown to the rewriter)
    chunks: list[dict]           # chunks from the latest search
    relevant: list[dict]         # chunks the grader judged useful
    started_at: float            # for the "researcher timings" table


# ---- Node: retrieve (no LLM) ---------------------------------------------
def retrieve(state: ResearcherState) -> dict:
    # First attempt: search with the sub-question itself.
    # Later attempts: search with the query written by `rewrite`.
    query = state.get("query") or state["sub_question"]
    chunks = search_documents(query)

    update = {
        "query": query,
        "chunks": chunks,
        "tried_queries": state.get("tried_queries", []) + [query],
    }
    if "started_at" not in state:  # first pass through this researcher
        update["started_at"] = time.time()
        update["rewrites"] = 0
        log.info("RESEARCHER START  t=%.3f  thread=%s  q=%r",
                 update["started_at"], threading.current_thread().name, state["sub_question"])
    log.info("  retrieve %d chunks for query=%r", len(chunks), query)
    return update


# ---- Node: grade (LLM decision #1: "are these chunks any good?") ---------
class Grade(BaseModel):
    """Structured output for the grader."""
    relevant: list[int] = Field(
        description="Numbers of the passages that help answer the question. "
                    "Empty list if none of them do."
    )


GRADER_PROMPT = """You check search results for a research assistant.
You get a question and numbered passages. Return the numbers of the passages
that contain information that helps answer the question. A passage that only
shares words with the question but not its meaning is NOT relevant.
If none are relevant, return an empty list."""


def grade(state: ResearcherState) -> dict:
    chunks = state["chunks"]
    if not chunks:  # empty database: nothing to grade, skip the LLM call
        return {"relevant": []}

    numbered = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks))
    result: Grade = get_llm("grader").with_structured_output(Grade).invoke([
        SystemMessage(GRADER_PROMPT),
        HumanMessage(f"Question: {state['sub_question']}\n\nPassages:\n\n{numbered}"),
    ])
    # Ignore any number outside the list (the LLM can get this wrong).
    relevant = [chunks[i] for i in dict.fromkeys(result.relevant) if 0 <= i < len(chunks)]
    log.info("  grade: %d/%d relevant for q=%r", len(relevant), len(chunks), state["sub_question"][:40])
    return {"relevant": relevant}


# ---- Conditional edge: a REAL condition, decided by the grade -----------
def route_after_grade(state: ResearcherState) -> str:
    if state["relevant"]:
        return "answer"                    # good chunks found
    if not state["chunks"]:
        return "answer"                    # database empty: rewriting can't help
    if state["rewrites"] < config.MAX_QUERY_REWRITES:
        return "rewrite"                   # try again with a better query
    return "answer"                        # out of retries: answer "not covered"


# ---- Node: rewrite (LLM decision #2: "how should I search instead?") -----
REWRITER_PROMPT = """You improve search queries for a document search engine
that matches by meaning. The previous queries did not find relevant passages.
Write ONE new query that uses different wording: technical terms, synonyms,
or the specific concept the question is really about.
Reply with the query only, no quotes, no explanation."""


def rewrite(state: ResearcherState) -> dict:
    tried = "\n".join(f"- {q}" for q in state["tried_queries"])
    response = get_llm("rewriter").invoke([
        SystemMessage(REWRITER_PROMPT),
        HumanMessage(f"Question: {state['sub_question']}\n\nQueries already tried:\n{tried}"),
    ])
    new_query = response.content.strip().strip('"').strip()
    log.info("  rewrite #%d: %r", state["rewrites"] + 1, new_query)

    # Send a progress event to the UI. The subgraph only exposes `findings`
    # to the outside, so its internal state (like `query`) is not visible in
    # the normal update stream; a custom stream event is how we report it.
    # (Does nothing when the graph is not being streamed with "custom".)
    get_stream_writer()({"event": "rewrite", "question": state["sub_question"], "query": new_query})
    return {"query": new_query, "rewrites": state["rewrites"] + 1}


# ---- Node: answer --------------------------------------------------------
ANSWER_PROMPT = """You are a meticulous research assistant answering ONE
sub-question of a larger research topic, using ONLY the passages given.

Citation rules (strict):
- Every claim must end with the bracketed tag of the passage it came from,
  copied exactly, e.g. [paper.pdf, p.3].
- Only use tags that appear in the passages. NEVER invent a citation.
- If the passages only partly answer the question, say what is missing.

Write a concise answer of 1-3 paragraphs."""

NOT_COVERED = "The uploaded documents do not contain information that answers this question."


def answer(state: ResearcherState) -> dict:
    relevant = state["relevant"]
    if relevant:
        passages = "\n\n".join(f"{c['tag']} {c['text']}" for c in relevant)
        response = get_llm("researcher").invoke([
            SystemMessage(ANSWER_PROMPT),
            HumanMessage(f"Overall topic: {state['topic']}\n\n"
                         f"Sub-question: {state['sub_question']}\n\nPassages:\n\n{passages}"),
        ])
        # Only tags of chunks the grader approved may appear as citations.
        allowed = {c["tag"] for c in relevant}
        text = strip_unknown_citations(response.content or "", allowed).strip()
        sources = [c for c in extract_citations(text) if c in allowed]
    else:
        text, sources = NOT_COVERED, []  # honest "not found" instead of guessing

    finished = time.time()
    log.info("RESEARCHER END    t=%.3f  (%.1fs, %d searches)  q=%r",
             finished, finished - state["started_at"], len(state["tried_queries"]),
             state["sub_question"])
    return {
        "findings": [{
            "question": state["sub_question"],
            "answer": text,
            "sources": sources,
            "queries": state["tried_queries"],  # shows the self-correction in the UI
            "started_at": state["started_at"],
            "finished_at": finished,
        }]
    }


def build_researcher():
    """Compile the self-correcting RAG subgraph (used as ONE node in the main graph)."""
    builder = StateGraph(
        ResearcherState,
        input_schema=ResearcherInput,    # accepts exactly what Send passes in
        output_schema=ResearcherOutput,  # returns only `findings` to the main graph
    )
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade)
    builder.add_node("rewrite", rewrite)
    builder.add_node("answer", answer)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    # The list names every node route_after_grade may return.
    builder.add_conditional_edges("grade", route_after_grade, ["answer", "rewrite"])
    builder.add_edge("rewrite", "retrieve")   # the self-correction loop
    builder.add_edge("answer", END)
    return builder.compile()  # no checkpointer: it inherits the main graph's


# ===========================================================================
# MAIN GRAPH
# ===========================================================================
class ResearchState(TypedDict):
    topic: str
    sub_questions: list[str]
    # REQUIRED reducer: several researchers finish in the same step and each
    # returns {"findings": [one_item]}. operator.add concatenates those lists.
    # Without it LangGraph would raise InvalidUpdateError (or lose findings).
    findings: Annotated[list, operator.add]
    draft: str


class Plan(BaseModel):
    """Structured output schema for the planner."""
    sub_questions: list[str] = Field(
        description=(
            f"{config.MIN_SUBQUESTIONS} to {config.MAX_SUBQUESTIONS} focused, "
            "non-overlapping research sub-questions"
        )
    )


# ---- Node: planner -------------------------------------------------------
PLANNER_PROMPT = f"""You are a research planner. Break the user's research topic
into {config.MIN_SUBQUESTIONS} to {config.MAX_SUBQUESTIONS} focused sub-questions.
- Each sub-question must be answerable on its own by one researcher.
- Sub-questions must not overlap; together they should cover the topic.
- Keep each one to a single, specific sentence."""


def planner(state: ResearchState) -> dict:
    llm = get_llm("planner").with_structured_output(Plan)
    plan: Plan = llm.invoke(
        [SystemMessage(PLANNER_PROMPT), HumanMessage(f"Research topic: {state['topic']}")]
    )
    questions = [q.strip() for q in plan.sub_questions if q.strip()]
    log.info("Planner produced %d sub-questions", len(questions))
    return {"sub_questions": questions[: config.MAX_SUBQUESTIONS]}


# ---- Node: approve_plan (human in the loop) ------------------------------
def approve_plan(state: ResearchState) -> dict:
    # interrupt() pauses the graph here and hands this payload to the caller.
    # When the caller runs graph.stream(Command(resume=value), same_config),
    # the node RE-RUNS from the top and interrupt() returns `value` instead.
    edited = interrupt({"sub_questions": state["sub_questions"]})

    # The app resumes with a list of lines; drop blank ones.
    questions = [q.strip() for q in edited if q.strip()]
    if not questions:  # user cleared everything: keep the original plan
        questions = state["sub_questions"]
    log.info("Plan approved with %d sub-questions: %s", len(questions), questions)
    return {"sub_questions": questions}


# ---- Edge: fan_out (one researcher per sub-question, in parallel) --------
def fan_out(state: ResearchState) -> list[Send]:
    # Not a condition: it always returns one Send per sub-question. Each Send
    # starts one researcher subgraph with its own small input dict.
    return [
        Send("researcher", {"topic": state["topic"], "sub_question": q})
        for q in state["sub_questions"]
    ]


# ---- Node: writer --------------------------------------------------------
WRITER_PROMPT = """You are a research writer. Merge the findings below into a
well-structured markdown report with exactly these parts:

# <A clear title for the report>
## Summary            (one short paragraph)
## <one section per sub-question, titled after it>
## Conclusion

Rules:
- Preserve EVERY citation tag exactly as written, e.g. [paper.pdf, p.3].
  Keep each one attached to the claim it supports.
- If a finding says the documents do not cover a question, say so in that
  section; do not fill the gap with outside knowledge.
- Do not add facts or citations that are not in the findings.
- Do NOT write a Sources section; it is appended automatically."""


def writer(state: ResearchState) -> dict:
    findings = state["findings"]
    findings_text = "\n\n".join(
        f"### Sub-question: {f['question']}\n{f['answer']}" for f in findings
    )
    response = get_llm("writer").invoke([
        SystemMessage(WRITER_PROMPT),
        HumanMessage(f"Topic: {state['topic']}\n\nFindings:\n\n{findings_text}"),
    ])

    # Every citation gathered by the researchers (ordered, no duplicates).
    # It is both the allow-list for the report and the Sources list.
    all_citations = list(dict.fromkeys(c for f in findings for c in f["sources"]))
    allowed = set(all_citations)
    body = strip_unknown_citations(response.content, allowed).strip()

    # Warn (in the log) if the writer dropped any citation from the findings.
    missing = allowed - set(extract_citations(body))
    if missing:
        log.warning("Writer dropped %d citation(s): %s", len(missing), sorted(missing))

    # The Sources list is built in code so it is always complete and correct.
    sources = "\n".join(f"- {c}" for c in all_citations) or "- (none)"
    draft = f"{body}\n\n## Sources\n\n{sources}\n"
    log.info("Writer finished (%d chars, %d sources)", len(draft), len(all_citations))
    return {"draft": draft}


# ---- Wiring --------------------------------------------------------------
def build_graph():
    builder = StateGraph(ResearchState)
    builder.add_node("planner", planner)
    builder.add_node("approve_plan", approve_plan)
    builder.add_node("researcher", build_researcher())  # a whole subgraph as one node
    builder.add_node("writer", writer)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "approve_plan")
    # fan_out returns a list of Send objects -> parallel researcher subgraphs.
    builder.add_conditional_edges("approve_plan", fan_out, ["researcher"])
    # writer waits until ALL researchers have finished.
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", END)

    # InMemorySaver keeps checkpoints in RAM, keyed by thread_id. Required for
    # interrupt/resume. (Use SqliteSaver to survive app restarts.)
    return builder.compile(checkpointer=InMemorySaver())


graph = build_graph()
