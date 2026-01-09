"""
Manga Ingestion Service

Fetches manga metadata from the Jikan API and stores raw JSON responses.
"""

from .main import (
    IngestionProgress,
    JikanClient,
    RateLimiter,
    clear_progress,
    get_ingestion_stats,
    ingest_manga,
)

__all__ = [
    "ingest_manga",
    "get_ingestion_stats",
    "clear_progress",
    "JikanClient",
    "RateLimiter",
    "IngestionProgress",
]
