"""
Manga Embedding Service

Converts manga metadata into searchable vector representations
using sentence-transformers for semantic search capabilities.
"""

from .main import (
    EmbeddingGenerator,
    EmbeddingStorage,
    MangaText,
    SemanticSearch,
    TextComposer,
    generate_embeddings,
    get_embedding_stats,
)

__all__ = [
    "EmbeddingGenerator",
    "EmbeddingStorage",
    "MangaText",
    "SemanticSearch",
    "TextComposer",
    "generate_embeddings",
    "get_embedding_stats",
]
