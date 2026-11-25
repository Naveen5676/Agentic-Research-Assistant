"""
retriever.py — search the indexed PDFs in ChromaDB.

This is a plain function, not an LLM tool: the researcher subgraph calls it
directly in its `retrieve` step. The LLM's job is to judge the results and,
if needed, write a better query (see graph.py).
"""

import config
from ingest import get_collection


def search_documents(query: str) -> list[dict]:
    """
    Return the TOP_K chunks closest in meaning to `query`.

    Chroma embeds the query text with the same model used at indexing time,
    then returns the nearest chunks by cosine distance.

    Each chunk is a dict:
        id   -> unique chunk id in Chroma (used to avoid duplicates)
        tag  -> citation tag, e.g. "[doc3.pdf, p.3]"
        text -> the chunk text with whitespace tidied up

    Returns [] when nothing is indexed.
    """
    result = get_collection().query(query_texts=[query], n_results=config.TOP_K)

    # Chroma returns one list per query text; we sent one, so take [0].
    return [
        {
            "id": chunk_id,
            "tag": f"[{meta.get('source', 'unknown')}, p.{meta.get('page', '?')}]",
            "text": " ".join(text.split()),  # collapse newlines/spaces
        }
        for chunk_id, text, meta in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0]
        )
    ]
