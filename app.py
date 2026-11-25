"""
app.py — Streamlit front end.

Run with:  streamlit run app.py

The UI has three stages, chosen from the graph's saved state:
  1. No run yet         -> enter a topic, "Start research"
  2. Interrupt pending  -> edit the planner's sub-questions, "Approve and research"
  3. Finished           -> show the report + download button

Key rule: the SAME thread_id must be used for the first invoke (which pauses
at the interrupt) and the resume. A new thread_id would start a fresh run.
"""

import uuid
from datetime import datetime

import groq
import streamlit as st
from langgraph.types import Command

import config
from graph import graph
from ingest import clear_collection, count_chunks, ingest_pdf

st.set_page_config(page_title="Agentic Research Assistant", page_icon="🔎", layout="wide")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
# Streamlit re-runs this whole script on every click, so anything that must
# survive between clicks goes in st.session_state.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "run_error" not in st.session_state:
    st.session_state.run_error = None  # message of the last failed run, if any


def run_config() -> dict:
    """The LangGraph config. Same thread_id across pause and resume."""
    return {
        "configurable": {"thread_id": st.session_state.thread_id},
        # Cap how many graph tasks (researchers) run at once, to stay under
        # Groq's tokens-per-minute limit. See config.MAX_CONCURRENT_RESEARCHERS.
        "max_concurrency": config.MAX_CONCURRENT_RESEARCHERS,
    }


def run_graph(graph_input, label: str) -> None:
    """
    Stream the graph until it pauses or finishes, showing progress, then
    record whether an interrupt is pending.

    graph_input can be: a dict (new run), Command(resume=...) (after the
    plan review), or None (retry: continue from the last checkpoint).
    """
    st.session_state.run_error = None
    with st.status(label, expanded=True) as status:
        try:
            for update in graph.stream(graph_input, run_config(), stream_mode="updates"):
                for node, payload in update.items():
                    if node == "__interrupt__":
                        status.write("⏸️ Waiting for your approval of the plan")
                    elif node == "researcher":
                        q = payload["findings"][0]["question"]
                        status.write(f"✅ Researcher finished: *{q}*")
                    else:
                        status.write(f"✅ {node} done")
        except Exception as exc:
            # Don't crash the page. The checkpoint keeps everything finished
            # so far, so the user can retry from where it stopped.
            if isinstance(exc, groq.RateLimitError):
                msg = ("Groq rate limit reached (free tier: 8,000 tokens/minute). "
                       "Wait about a minute, then click **Retry**.")
            else:
                msg = f"The run failed: `{type(exc).__name__}: {exc}`"
            st.session_state.run_error = msg
            st.session_state.interrupt_payload = None
            status.update(label="Stopped with an error", state="error", expanded=False)
            return
        status.update(label="Done", state="complete", expanded=False)

    # Detect the pause by reading the checkpointed state (as recommended):
    # a pending interrupt shows up on the next task to run.
    snapshot = graph.get_state(run_config())
    if snapshot.tasks and snapshot.tasks[0].interrupts:
        st.session_state.interrupt_payload = snapshot.tasks[0].interrupts[0].value
    else:
        st.session_state.interrupt_payload = None


# ---------------------------------------------------------------------------
# Sidebar: documents + reset
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Documents")
    uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if st.button("Index documents", disabled=not uploads, use_container_width=True):
        total = 0
        with st.spinner("Embedding and indexing..."):
            for f in uploads:
                added = ingest_pdf(f.getvalue(), f.name)
                total += added
                st.write(f"{f.name}: {added} chunks" + (" (already indexed)" if added == 0 else ""))
        st.success(f"Indexed {total} new chunks.")

    # Two-step delete: the button only unlocks after ticking the checkbox,
    # so one stray click can't wipe the index.
    confirm = st.checkbox("Yes, delete all indexed documents")
    if st.button("🗑️ Clear all documents", disabled=not confirm, use_container_width=True):
        removed = clear_collection()
        st.success(f"Removed {removed} chunks. Upload and index documents again.")

    # Drawn after the buttons so it always shows the up-to-date number.
    st.caption(f"📦 Chunks in database: **{count_chunks()}**")

    st.divider()
    if st.button("Reset run", use_container_width=True):
        # A fresh thread_id = a brand-new run with empty state.
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.interrupt_payload = None
        st.session_state.run_error = None
        st.rerun()
    st.caption(f"Thread: `{st.session_state.thread_id[:8]}`")


# ---------------------------------------------------------------------------
# Main area: pick the stage from saved graph state
# ---------------------------------------------------------------------------
st.title("🔎 Agentic Research Assistant")

state_values = graph.get_state(run_config()).values
draft = state_values.get("draft")

# ---- Run stopped with an error (e.g. rate limit) -------------------------
if st.session_state.run_error:
    st.error(st.session_state.run_error)
    st.caption("Retry continues from the last checkpoint: researchers that "
               "already finished are not run again.")
    if st.button("Retry", type="primary"):
        # None as input = "resume this thread from its last checkpoint".
        run_graph(None, "Retrying...")
        st.rerun()

# ---- Stage 2: plan review ------------------------------------------------
elif st.session_state.interrupt_payload:
    st.subheader("Review the research plan")
    st.write(f"**Topic:** {state_values.get('topic', '')}")
    st.caption("One sub-question per line. Edit, add, or delete lines, then approve.")

    plan_text = st.text_area(
        "Sub-questions",
        value="\n".join(st.session_state.interrupt_payload["sub_questions"]),
        height=200,
    )
    edited = [line.strip() for line in plan_text.splitlines() if line.strip()]

    if st.button("Approve and research", type="primary", disabled=not edited):
        # Resume the paused graph; the list becomes interrupt()'s return value.
        run_graph(Command(resume=edited), f"Researching {len(edited)} sub-questions in parallel...")
        st.rerun()

# ---- Stage 3: finished report -------------------------------------------
elif draft:
    st.download_button("⬇️ Download report.md", draft, file_name="report.md", mime="text/markdown")

    # Show how the researchers overlapped in time (proof they ran in parallel).
    findings = state_values.get("findings", [])
    if findings:
        with st.expander("Researcher timings"):
            t0 = min(f["started_at"] for f in findings)
            rows = [{
                "sub-question": f["question"],
                "start (s)": round(f["started_at"] - t0, 2),
                "end (s)": round(f["finished_at"] - t0, 2),
                "started at": datetime.fromtimestamp(f["started_at"]).strftime("%H:%M:%S.%f")[:-3],
                "tool calls": f["tool_calls"],
            } for f in findings]
            st.dataframe(rows, use_container_width=True)

    st.markdown(draft)

# ---- Stage 1: no run yet --------------------------------------------------
else:
    topic = st.text_input("Research topic", placeholder="e.g. How do retrieval-augmented LLMs reduce hallucination?")

    # Sample topic for testing. st.code shows a copy button on hover.
    st.caption("Sample topic for testing (click the copy icon on the right):")
    st.code(
        "How should a retrieval system be designed for a large collection of technical documentation?",
        language=None,
        wrap_lines=True,
    )

    if st.button("Start research", type="primary", disabled=not topic.strip()):
        run_graph({"topic": topic.strip()}, "Planning sub-questions...")
        st.rerun()
