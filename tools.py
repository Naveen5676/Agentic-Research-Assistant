"""
tools.py — the two tools researcher agents can call.

Both tools return plain strings in which every passage starts with a
bracketed source tag. The researcher is told to copy those tags verbatim as
citations, so every citation in the final report comes from a real tool
result:
    search_documents -> "[paper.pdf, p.3] ...passage..."
    search_web       -> "[https://example.com/page] ...content..."

The LLM decides which tool to call (and when) from the docstrings below,
which are sent to the model as the tool descriptions.
"""

from functools import lru_cache

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

import config
from ingest import get_vectorstore


@tool
def search_documents(query: str) -> str:
    """Search the user's uploaded PDF documents. Use this for anything the
    uploaded papers could cover: their methods, results, definitions, data.
    Returns passages tagged with [filename, p.N]."""
    hits = get_vectorstore().similarity_search(query, k=config.TOP_K)
    if not hits:
        return "No relevant passages found."

    passages = []
    for doc in hits:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        text = " ".join(doc.page_content.split())  # collapse newlines/spaces
        passages.append(f"[{source}, p.{page}] {text}")
    return "\n\n".join(passages)


@lru_cache(maxsize=1)
def _tavily() -> TavilySearch:
    """Create the Tavily client once (reads TAVILY_API_KEY from the env)."""
    return TavilySearch(max_results=config.WEB_MAX_RESULTS)


@tool
def search_web(query: str) -> str:
    """Search the live web. Use this for current events, recent developments,
    or information external to the uploaded documents.
    Returns snippets tagged with [url]."""
    try:
        response = _tavily().invoke({"query": query})
    except Exception as exc:  # missing key, network error, rate limit, ...
        # Return the error as text so the agent can carry on with the other
        # tool instead of crashing the whole run.
        return f"Web search failed: {exc}"

    results = response.get("results", []) if isinstance(response, dict) else []
    if not results:
        return "No web results found."
    return "\n\n".join(f"[{r['url']}] {r.get('content', '').strip()}" for r in results)


# Handy list for binding to a model.
TOOLS = [search_documents, search_web]
