"""
graph.py — the LangGraph workflow.

    START
      -> planner        (LLM splits topic into 3-5 sub-questions)
      -> approve_plan   (interrupt: the human edits / approves the plan)
      -> fan_out        (conditional edge: one Send per sub-question)
      -> researcher xN  (run IN PARALLEL, each a small ReAct tool loop)
      -> writer         (merges findings into a cited markdown report)
    -> END

The graph is compiled with an InMemorySaver checkpointer, which is what lets
interrupt() pause the run and Command(resume=...) continue it later on the
same thread_id.
"""

import logging
import operator
import re
import threading
import time
from functools import lru_cache
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt
from pydantic import BaseModel, Field

import config
from tools import TOOLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(threadName)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("research_graph")
# Silence chatty HTTP / search-library loggers so our START/END lines stand out.
for noisy in ("httpx", "sentence_transformers", "chromadb"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class ResearchState(TypedDict):
    topic: str
    sub_questions: list[str]
    # REQUIRED reducer: several researchers finish in the same step and each
    # returns {"findings": [one_item]}. operator.add concatenates those lists.
    # Without it LangGraph would raise InvalidUpdateError (or lose findings).
    findings: Annotated[list, operator.add]
    draft: str


class ResearcherInput(TypedDict):
    """What a single researcher receives via Send — NOT the full state."""
    topic: str
    sub_question: str


class Plan(BaseModel):
    """Structured output schema for the planner."""
    sub_questions: list[str] = Field(
        description=(
            f"{config.MIN_SUBQUESTIONS} to {config.MAX_SUBQUESTIONS} focused, "
            "non-overlapping research sub-questions"
        )
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


# Citation patterns produced by tools.py:
#   [paper.pdf, p.3]   and   [https://example.com/x]
DOC_CITATION = re.compile(r"\[[^\[\]\n]+?, p\.\d+\]")
WEB_CITATION = re.compile(r"\[https?://[^\[\]\s]+\]")


def extract_citations(text: str) -> list[str]:
    """All citation tags in `text`, in order of first appearance, no duplicates."""
    found = DOC_CITATION.findall(text) + WEB_CITATION.findall(text)
    return list(dict.fromkeys(sorted(found, key=text.index)))


def strip_unknown_citations(text: str, allowed: set[str]) -> str:
    """
    Remove any citation-looking tag that did not come from a tool result.
    This is the hard guarantee behind "never invent a citation".
    """
    def keep_or_drop(match: re.Match) -> str:
        tag = match.group(0)
        if tag in allowed:
            return tag
        log.warning("Removed invented citation: %s", tag)
        return ""

    text = DOC_CITATION.sub(keep_or_drop, text)
    return WEB_CITATION.sub(keep_or_drop, text)


# ---------------------------------------------------------------------------
# Node 1: planner
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Node 2: approve_plan (human in the loop)
# ---------------------------------------------------------------------------
def approve_plan(state: ResearchState) -> dict:
    # interrupt() pauses the graph here and hands this payload to the caller.
    # When the caller runs graph.invoke(Command(resume=value), same_config),
    # the node RE-RUNS from the top and interrupt() returns `value` instead.
    edited = interrupt({"sub_questions": state["sub_questions"]})

    # The app resumes with a list of lines; drop blank ones.
    questions = [q.strip() for q in edited if q.strip()]
    if not questions:  # user cleared everything: keep the original plan
        questions = state["sub_questions"]
    log.info("Plan approved with %d sub-questions: %s", len(questions), questions)
    return {"sub_questions": questions}


# ---------------------------------------------------------------------------
# Edge: fan_out (conditional edge -> parallel researchers)
# ---------------------------------------------------------------------------
def fan_out(state: ResearchState) -> list[Send]:
    # One Send per sub-question. All researchers run in the same superstep,
    # so LangGraph executes them concurrently.
    return [
        Send("researcher", {"topic": state["topic"], "sub_question": q})
        for q in state["sub_questions"]
    ]


# ---------------------------------------------------------------------------
# Node 3: researcher (small ReAct loop)
# ---------------------------------------------------------------------------
RESEARCHER_PROMPT = f"""You are a meticulous research assistant answering ONE
sub-question of a larger research topic.

Tools:
- search_documents: use for anything the user's uploaded papers could cover.
- search_web: use for current events or information external to the papers.
Call one or both as needed (at most {config.MAX_TOOL_CALLS} calls in total).
Usually try search_documents first.

Citation rules (strict):
- Every claim must end with the bracketed source tag exactly as the tool
  returned it, e.g. [paper.pdf, p.3] or [https://example.com/page].
- Only cite tags that appear in tool results. NEVER invent a citation.
- If the tools found nothing useful, say so plainly instead of guessing.

Finish with a concise answer of 1-3 paragraphs."""

TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def researcher(inputs: ResearcherInput) -> dict:
    question = inputs["sub_question"]
    started = time.time()
    log.info("RESEARCHER START  t=%.3f  thread=%s  q=%r",
             started, threading.current_thread().name, question)

    llm_with_tools = get_llm("researcher").bind_tools(TOOLS)
    llm_plain = get_llm("researcher")  # used for the final answer once the cap is hit

    messages = [
        SystemMessage(RESEARCHER_PROMPT),
        HumanMessage(f"Overall topic: {inputs['topic']}\n\nSub-question: {question}"),
    ]
    tool_calls_used = 0
    tool_outputs: list[str] = []

    while True:
        # Once the budget is spent, call the model without tools so it must answer.
        llm = llm_with_tools if tool_calls_used < config.MAX_TOOL_CALLS else llm_plain
        response: AIMessage = llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:  # model produced its final answer
            break

        for call in response.tool_calls:
            # Every tool call needs a ToolMessage reply, even refused ones.
            if tool_calls_used >= config.MAX_TOOL_CALLS:
                content = "Tool call limit reached. Answer with the information you have."
            else:
                tool_calls_used += 1
                tool = TOOLS_BY_NAME.get(call["name"])
                content = (
                    tool.invoke(call["args"]) if tool
                    else f"Unknown tool: {call['name']}"
                )
                tool_outputs.append(content)
                log.info("  tool #%d %s(%r) for q=%r",
                         tool_calls_used, call["name"], call["args"], question[:40])
            messages.append(ToolMessage(content=content, tool_call_id=call["id"]))

    # Only tags that really appeared in tool output are allowed as citations.
    allowed = set(extract_citations("\n".join(tool_outputs)))
    answer = strip_unknown_citations(response.content or "", allowed).strip()
    sources = [c for c in extract_citations(answer) if c in allowed]

    finished = time.time()
    log.info("RESEARCHER END    t=%.3f  (%.1fs, %d tool calls)  q=%r",
             finished, finished - started, tool_calls_used, question)

    return {
        "findings": [{
            "question": question,
            "answer": answer,
            "sources": sources,
            # Timing kept so the UI can show the researchers overlapped.
            "started_at": started,
            "finished_at": finished,
            "tool_calls": tool_calls_used,
        }]
    }


# ---------------------------------------------------------------------------
# Node 4: writer
# ---------------------------------------------------------------------------
WRITER_PROMPT = """You are a research writer. Merge the findings below into a
well-structured markdown report with exactly these parts:

# <A clear title for the report>
## Summary            (one short paragraph)
## <one section per sub-question, titled after it>
## Conclusion

Rules:
- Preserve EVERY citation tag exactly as written, e.g. [paper.pdf, p.3] or
  [https://example.com]. Keep each one attached to the claim it supports.
- Do not add facts or citations that are not in the findings.
- Do NOT write a Sources section; it is appended automatically."""


def _sources_section(citations: list[str]) -> str:
    """Build the Sources list in code so it is always complete and correct."""
    docs = [c for c in citations if DOC_CITATION.fullmatch(c)]
    webs = [c for c in citations if WEB_CITATION.fullmatch(c)]
    lines = ["## Sources", "", "### Documents"]
    lines += [f"- {c}" for c in docs] or ["- (none)"]
    lines += ["", "### Web"]
    # Strip the brackets so the URL renders as a clickable link.
    lines += [f"- [{c[1:-1]}]({c[1:-1]})" for c in webs] or ["- (none)"]
    return "\n".join(lines)


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

    draft = f"{body}\n\n{_sources_section(all_citations)}\n"
    log.info("Writer finished (%d chars, %d sources)", len(draft), len(all_citations))
    return {"draft": draft}


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(ResearchState)
    builder.add_node("planner", planner)
    builder.add_node("approve_plan", approve_plan)
    builder.add_node("researcher", researcher)
    builder.add_node("writer", writer)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "approve_plan")
    # fan_out returns a list of Send objects -> parallel researcher tasks.
    builder.add_conditional_edges("approve_plan", fan_out, ["researcher"])
    # writer waits until ALL researchers of the superstep have finished.
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", END)

    # InMemorySaver keeps checkpoints in RAM, keyed by thread_id. Required for
    # interrupt/resume. (Use SqliteSaver to survive app restarts.)
    return builder.compile(checkpointer=InMemorySaver())


graph = build_graph()
