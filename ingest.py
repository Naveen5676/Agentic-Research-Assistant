"""
ingest.py — turn PDFs into searchable chunks stored in ChromaDB.

Pipeline for each PDF:
    bytes --SHA-256--> skip if already indexed
          --PyPDFLoader--> one Document per page
          --RecursiveCharacterTextSplitter--> ~1000-char chunks
          --embeddings--> stored in the Chroma collection
"""

import hashlib
import os
import tempfile
import threading
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the local embedding model once and reuse it (loading is slow)."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        # Unit-length vectors make cosine similarity well behaved.
        encode_kwargs={"normalize_embeddings": True},
    )


_store_lock = threading.Lock()


def get_vectorstore() -> Chroma:
    """Thread-safe accessor: parallel researchers must not build it twice."""
    with _store_lock:
        return _open_vectorstore()


@lru_cache(maxsize=1)
def _open_vectorstore() -> Chroma:
    """
    Open (or create) the persistent Chroma collection.

    Distance metric: Chroma defaults to squared L2 and the metric CANNOT be
    changed once a collection exists, so we set cosine at creation time.
    The installed chromadb (1.x) takes this through the `configuration`
    argument ({"hnsw": {"space": "cosine"}}); the old
    metadata={"hnsw:space": "cosine"} form is the pre-1.0 way to do it.
    """
    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=config.PERSIST_DIR,
        collection_configuration={"hnsw": {"space": "cosine"}},
    )

    # Safety check: an old collection created with L2 would silently give
    # worse rankings. Fail loudly instead so the user can delete chroma_db/.
    space = store._collection.configuration.get("hnsw", {}).get("space")
    if space != "cosine":
        raise RuntimeError(
            f"Chroma collection '{config.COLLECTION_NAME}' uses distance "
            f"'{space}', expected 'cosine'. Delete {config.PERSIST_DIR} and re-index."
        )
    return store


def _splitter() -> RecursiveCharacterTextSplitter:
    """Splitter that prefers paragraph, then line, then sentence boundaries."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _already_indexed(store: Chroma, file_hash: str) -> bool:
    """True if any chunk with this file hash is already in the collection."""
    hits = store.get(where={"file_hash": file_hash}, limit=1)
    return len(hits["ids"]) > 0


def ingest_pdf(file_bytes: bytes, filename: str) -> int:
    """
    Index one PDF given its raw bytes. Returns the number of chunks added
    (0 if the exact same file was indexed before).
    """
    store = get_vectorstore()

    # Hash the bytes, not the name: renaming a file won't re-index it, and
    # editing a file with the same name will.
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    if _already_indexed(store, file_hash):
        return 0

    # PyPDFLoader needs a path on disk, so write the bytes to a temp file.
    # delete=False + manual cleanup because Windows cannot reopen an open temp file.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        pages = PyPDFLoader(tmp_path).load()
    finally:
        os.remove(tmp_path)

    chunks = _splitter().split_documents(pages)
    chunks = [c for c in chunks if c.page_content.strip()]  # drop empty pages
    if not chunks:
        return 0

    # Replace the loader's metadata with exactly what we need for citations.
    for chunk in chunks:
        chunk.metadata = {
            "source": filename,
            # PyPDFLoader pages are 0-based; humans count from 1.
            "page": int(chunk.metadata.get("page", 0)) + 1,
            "file_hash": file_hash,
        }

    # Deterministic IDs so a retried ingest can never create duplicates.
    ids = [f"{file_hash[:16]}-{i}" for i in range(len(chunks))]
    store.add_documents(chunks, ids=ids)
    return len(chunks)


def count_chunks() -> int:
    """How many chunks are currently stored (shown in the sidebar)."""
    return get_vectorstore()._collection.count()


def clear_collection() -> int:
    """
    Delete EVERY indexed chunk by dropping the whole Chroma collection.
    Returns how many chunks were removed.

    The collection is recreated (with cosine distance) the next time
    get_vectorstore() is called, so indexing works again right away.
    """
    with _store_lock:
        store = _open_vectorstore()
        removed = store._collection.count()
        store.delete_collection()
        # Forget the cached handle: it points at the deleted collection.
        _open_vectorstore.cache_clear()
    return removed


def ingest_path(path: str | Path) -> int:
    """Convenience wrapper: index a PDF that is already on disk."""
    path = Path(path)
    return ingest_pdf(path.read_bytes(), path.name)


if __name__ == "__main__":
    # Usage: python ingest.py paper1.pdf paper2.pdf
    import sys

    for p in sys.argv[1:]:
        print(f"{p}: {ingest_path(p)} chunks added")
