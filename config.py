"""
config.py — every tunable value for the app lives here.

Change models, temperatures, or retrieval settings in this one file instead of
hunting through the code.
"""

import os

from dotenv import load_dotenv

# Load variables from the .env file (GROQ_API_KEY, TAVILY_API_KEY, ...) into
# os.environ. Every other module imports config first, so this runs once, early.
load_dotenv()

# ---------------------------------------------------------------------------
# LLM (served by Groq)
# ---------------------------------------------------------------------------
# "groq:<model>" tells langchain.chat_models.init_chat_model to use the
# langchain-groq integration. "openai/gpt-oss-120b" is OpenAI's open-weight
# GPT model hosted on Groq. Swap to "groq:openai/gpt-oss-20b" for a faster,
# cheaper (but weaker) model.
MODEL = os.getenv("MODEL", "groq:openai/gpt-oss-120b")

# One temperature per agent role. Planner and researchers should be
# deterministic; the writer gets a little creativity for nicer prose.
TEMPERATURES = {
    "planner": 0.0,
    "researcher": 0.0,
    "writer": 0.6,
}

# ---------------------------------------------------------------------------
# Embeddings (local, free — Groq does not offer an embeddings API)
# ---------------------------------------------------------------------------
# A small sentence-transformers model that runs on CPU. It is downloaded once
# (~90 MB) on first use and cached by Hugging Face afterwards.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Document ingestion and retrieval
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1000          # characters per chunk
CHUNK_OVERLAP = 150        # characters shared between neighbouring chunks
TOP_K = 4                  # chunks returned by each search_documents call

# ---------------------------------------------------------------------------
# Agent behaviour
# ---------------------------------------------------------------------------
MIN_SUBQUESTIONS = 3
MAX_SUBQUESTIONS = 5
MAX_TOOL_CALLS = 4         # hard cap on tool calls per researcher
WEB_MAX_RESULTS = 3        # results per web search

# ---------------------------------------------------------------------------
# Rate-limit protection (Groq free tier = 8,000 tokens per minute)
# ---------------------------------------------------------------------------
# How many researchers may run at the same moment. Still parallel, but
# 2 at a time instead of all 5 keeps token bursts under the per-minute limit.
# Raise this if you upgrade your Groq plan.
MAX_CONCURRENT_RESEARCHERS = 2

# On a 429 "rate limit" error the Groq client waits the time Groq asks for
# (the retry-after header) and tries again, up to this many times.
MAX_RETRIES = 6

# gpt-oss models "think" before answering and those hidden tokens count
# toward the limit. "low" thinks less = far fewer tokens. ("medium"/"high")
REASONING_EFFORT = "low"

# ---------------------------------------------------------------------------
# ChromaDB storage
# ---------------------------------------------------------------------------
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "research_docs"
