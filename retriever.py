"""
retriever.py — search the indexed PDFs in ChromaDB.

This is a plain function, not an LLM tool: the researcher subgraph calls it
directly in its `retrieve` step. The LLM's job is to judge the results and,
if needed, write a better query (see graph.py).
"""

import config
from ingest import get_vectorstore


def search_documents(query: str) -> list[dict]:
    """
    Return the TOP_K chunks closest in meaning to `query`.

    Each chunk is a dict:
        id   -> unique chunk id in Chroma (used to avoid duplicates)
        tag  -> citation tag, e.g. "[doc3.pdf, p.3]"
        text -> the chunk text with whitespace tidied up

    Returns [] when nothing is indexed.
    """
    hits = get_vectorstore().similarity_search(query, k=config.TOP_K)
    return [
        {
            "id": doc.id,
            "tag": f"[{doc.metadata.get('source', 'unknown')}, p.{doc.metadata.get('page', '?')}]",
            "text": " ".join(doc.page_content.split()),  # collapse newlines/spaces
        }
        for doc in hits
    ]
