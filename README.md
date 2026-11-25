# Agentic Research Assistant

Upload PDFs, enter a research topic, and get a cited markdown report.

1. A **planner** agent splits the topic into 3–5 sub-questions.
2. **You review and edit** the plan (LangGraph `interrupt`).
3. **Researcher** agents run **in parallel**, one per sub-question. Each one picks
   between your uploaded documents and live web search.
4. A **writer** agent merges the findings into a report where every claim is cited
   as `[filename, p.N]` or `[url]`.

**Stack:** LangGraph · LangChain · Groq (`openai/gpt-oss-120b`) · ChromaDB ·
local sentence-transformers embeddings · Tavily web search · Streamlit

## Setup

Requires Python 3.10–3.13. Python 3.14 may not have wheels for every dependency yet.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API keys
copy .env.example .env          # Windows   (cp on macOS / Linux)
# then edit .env

# 4. Run
streamlit run app.py
```

The first time you index documents, the embedding model (~90 MB) downloads once.

## Environment variables

| Variable         | Required | Purpose                                                                 |
|------------------|----------|-------------------------------------------------------------------------|
| `GROQ_API_KEY`   | yes      | LLM calls. Get one at https://console.groq.com/keys                     |
| `TAVILY_API_KEY` | yes      | Web search. Get one at https://app.tavily.com                         |
| `MODEL`          | no       | Override the model, e.g. `groq:openai/gpt-oss-20b`                      |

Embeddings run locally, so no OpenAI key is needed. Groq has no embeddings API.

## Using the app

1. **Sidebar:** upload PDFs, then click **Index documents**. Files you already
   indexed are skipped (detected by SHA-256 of the file bytes).
2. Enter a topic and click **Start research**.
3. Edit the sub-questions (one per line), then click **Approve and research**.
4. Read the report, or download it as `report.md`. The **Researcher timings**
   panel shows that the researchers ran at the same time.
5. **Reset run** starts a new research thread.

Researcher start and end times are also logged to the terminal:

```
19:27:54.914 [ThreadPoolExecutor-3_0] RESEARCHER START ...
19:27:54.914 [ThreadPoolExecutor-3_1] RESEARCHER START ...
```

You can also index PDFs from the command line: `python ingest.py a.pdf b.pdf`

## Files

| File        | What it does                                                              |
|-------------|---------------------------------------------------------------------------|
| `config.py` | Model name, per-role temperatures, chunking and retrieval constants      |
| `ingest.py` | PDF loading, chunking, SHA-256 dedup, Chroma collection (cosine distance) |
| `tools.py`  | `search_documents` and `search_web` tools, which return cited passages   |
| `graph.py`  | LangGraph state, nodes, parallel fan-out, compiled graph                 |
| `app.py`    | Streamlit UI and the interrupt/resume cycle                              |

## How citations are enforced

- The tools prefix every passage with its source tag.
- Researchers are told to copy those tags exactly.
- In code, any citation-shaped tag that did not appear in a tool result is
  **removed**, both in researcher answers and in the final report.
- The **Sources** section is built in code from the citations the researchers
  collected. It lists documents and web links separately.

## Notes

- The Chroma collection uses **cosine** distance. The metric can't be changed
  after the collection is created, so if you see an error about the distance
  metric, delete `chroma_db/` and index again.
- Run state lives in memory (`InMemorySaver`). Restarting the app clears runs
  that are in progress, but indexed documents stay in `chroma_db/`.
