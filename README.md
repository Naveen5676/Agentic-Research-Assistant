# Agentic Research Assistant

Upload PDFs, enter a research topic, and get a cited markdown report built
only from your documents.

1. A **planner** agent splits the topic into 3–5 sub-questions.
2. **You review and edit** the plan (LangGraph `interrupt`).
3. **Researcher** agents run **in parallel**, one per sub-question. Each one is
   a **self-correcting RAG** subgraph: it searches your documents, has the LLM
   grade the results, and rewrites its query and searches again if nothing
   relevant came back.
4. A **writer** agent merges the findings into a report where every claim is
   cited as `[filename, p.N]`.

**Stack:** LangGraph · LangChain · Groq (`openai/gpt-oss-120b`) · ChromaDB ·
local sentence-transformers embeddings · Streamlit

## How it works

```
MAIN GRAPH
START → planner → approve_plan ──(fan_out: one Send per sub-question)──→ researcher ×N → writer → END
                    ⏸ you edit the plan                                    (parallel)

RESEARCHER SUBGRAPH (one copy per sub-question)
START → retrieve → grade ──(route_after_grade)──→ answer → END
           ↑                       │
           └─────── rewrite ←──────┘   (no relevant chunks, rewrites left)
```

| Step | LLM? | What it does |
|---|---|---|
| `retrieve` | no | Searches Chroma with the current query (the sub-question at first) |
| `grade` | yes | Picks which of the returned chunks actually help answer the question |
| `route_after_grade` | no | Relevant chunks → `answer`. None, with rewrites left → `rewrite`. None, no rewrites left → `answer` |
| `rewrite` | yes | Writes a better search query, told which queries already failed |
| `answer` | yes | Answers from the relevant chunks only, citing each claim. If nothing relevant was found, says the documents don't cover it |

`MAX_QUERY_REWRITES` (default 2, so at most 3 searches) is a hard limit, so a
researcher can never loop forever.

## Setup

Requires Python 3.10–3.13. Python 3.14 may not have wheels for every dependency yet.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
copy .env.example .env          # Windows   (cp on macOS / Linux)
# then edit .env

# 4. Run
streamlit run app.py
```

The first time you index documents, the embedding model (~90 MB) downloads once.

## Environment variables

| Variable       | Required | Purpose                                              |
|----------------|----------|------------------------------------------------------|
| `GROQ_API_KEY` | yes      | LLM calls. Get one at https://console.groq.com/keys  |
| `MODEL`        | no       | Override the model, e.g. `groq:openai/gpt-oss-20b`   |

Embeddings run locally, so no OpenAI key is needed. Groq has no embeddings API.

## Using the app

1. **Sidebar:** upload PDFs, then click **Index documents**. Files you already
   indexed are skipped (detected by SHA-256 of the file bytes). To start over,
   tick the confirmation box and click **Clear all documents**.
2. Enter a topic and click **Start research**.
3. Edit the sub-questions (one per line), then click **Approve and research**.
   The progress box shows a 🔁 line each time a researcher rewrites its query.
4. Read the report, or download it as `report.md`. The **Researcher details**
   panel shows that the researchers ran at the same time and which queries
   each one used.
5. **Reset run** starts a new research thread.

Each researcher step is also logged to the terminal:

```
10:02:11.410 [ThreadPoolExecutor-3_0] RESEARCHER START ...
10:02:11.410 [ThreadPoolExecutor-3_1] RESEARCHER START ...
10:02:12.902 [ThreadPoolExecutor-3_1]   grade: 0/4 relevant for q='...'
10:02:13.655 [ThreadPoolExecutor-3_1]   rewrite #1: 'reciprocal rank fusion constant k'
```

You can also index PDFs from the command line: `python ingest.py a.pdf b.pdf`

## Files

| File           | What it does                                                              |
|----------------|---------------------------------------------------------------------------|
| `config.py`    | Model name, per-role temperatures, chunking, retry and rate-limit settings |
| `ingest.py`    | PDF loading, chunking, SHA-256 dedup, Chroma collection (cosine distance) |
| `retriever.py` | `search_documents`: returns the closest chunks with their citation tags  |
| `graph.py`     | Main graph, researcher subgraph, state, nodes, compiled graph            |
| `app.py`       | Streamlit UI and the interrupt/resume cycle                              |

## How citations are enforced

- Every retrieved chunk carries its tag, e.g. `[doc3.pdf, p.3]`.
- The researcher only sees chunks the grader approved, and is told to copy
  their tags exactly.
- In code, any citation tag that is not from an approved chunk is **removed**,
  both in researcher answers and in the final report.
- The **Sources** section is built in code from the citations the researchers
  collected.

## Groq free tier

The free tier allows 8,000 tokens per minute. To stay under it, at most
`MAX_CONCURRENT_RESEARCHERS` (default 2) researchers run at once, gpt-oss runs
with low reasoning effort, and rate-limit errors are retried automatically. If
a run still fails, the app shows a **Retry** button that continues from the
last checkpoint.

## Notes

- The Chroma collection uses **cosine** distance. The metric can't be changed
  after the collection is created, so if you see an error about the distance
  metric, delete `chroma_db/` and index again.
- Run state lives in memory (`InMemorySaver`). Restarting the app clears runs
  that are in progress, but indexed documents stay in `chroma_db/`.
