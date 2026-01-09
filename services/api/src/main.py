"""
MangaHelp Search API

FastAPI backend for semantic manga search using pre-computed embeddings.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configuration
# Paths - check for Docker mount first, then fall back to project-relative paths
# In Docker, data is mounted at /app/data
# Locally, data is at PROJECT_ROOT/data
if Path("/app/data").exists():
    DATA_DIR = Path("/app/data")
else:
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"

EMBEDDINGS_DIR = DATA_DIR / "embeddings"
RAW_DIR = DATA_DIR / "raw"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="MangaHelp Search API",
    description="Semantic search for manga using natural language queries",
    version="1.0.0",
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class MangaResult(BaseModel):
    """A single manga search result."""

    mal_id: int = Field(..., description="MyAnimeList ID")
    title: str = Field(..., description="Manga title")
    score: float = Field(..., description="Similarity score (0-1)")
    genres: List[str] = Field(default_factory=list, description="Genre list")
    themes: List[str] = Field(default_factory=list, description="Theme list")
    demographics: List[str] = Field(default_factory=list, description="Demographics")
    synopsis: Optional[str] = Field(None, description="Plot synopsis")
    image_url: Optional[str] = Field(None, description="Cover image URL")
    mal_score: Optional[float] = Field(None, description="MAL user score")
    url: Optional[str] = Field(None, description="MAL URL")
    popularity: Optional[int] = Field(None, description="Popularity rank")
    members: Optional[int] = Field(None, description="Number of members")


class SearchResponse(BaseModel):
    """Response for search endpoint."""

    query: str
    results: List[MangaResult]
    total_results: int
    search_time_ms: float


class StatsResponse(BaseModel):
    """Response for stats endpoint."""

    manga_count: int
    embedding_dimension: int
    has_embeddings: bool
    detail_coverage: float


class GenreInfo(BaseModel):
    """Genre information."""

    mal_id: int
    name: str
    count: int = Field(..., description="Number of manga with this genre")


class GenresResponse(BaseModel):
    """Response for genres endpoint."""

    genres: List[GenreInfo]
    total_genres: int


class TopMangaResponse(BaseModel):
    """Response for top manga endpoint."""

    manga: List[MangaResult]
    total: int


# Global state for loaded data
class SearchEngine:
    """Manages embeddings and search functionality."""

    def __init__(self):
        self.embeddings: Optional[np.ndarray] = None
        self.normalized_embeddings: Optional[np.ndarray] = None
        self.mal_ids: Optional[np.ndarray] = None
        self.metadata: Optional[List[dict]] = None
        self.manga_data: dict = {}  # Full manga data by mal_id
        self.model = None
        self.is_loaded = False
        self.genres: dict = {}  # genre_name -> {mal_id, name, count}
        self.manga_by_popularity: List[int] = []  # mal_ids sorted by popularity

    def load(self):
        """Load embeddings and manga data."""
        if self.is_loaded:
            return

        logger.info("Loading search engine data...")

        # Load embeddings
        embeddings_file = EMBEDDINGS_DIR / "embeddings_latest.npy"
        mal_ids_file = EMBEDDINGS_DIR / "mal_ids_latest.npy"
        metadata_file = EMBEDDINGS_DIR / "metadata_latest.json"

        if not embeddings_file.exists():
            raise FileNotFoundError("No embeddings found. Run embedding service first.")

        self.embeddings = np.load(embeddings_file)
        self.mal_ids = np.load(mal_ids_file)

        with open(metadata_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.normalized_embeddings = self.embeddings / np.maximum(norms, 1e-10)

        logger.info(f"Loaded {len(self.mal_ids)} manga embeddings")

        # Load full manga data for richer results
        self._load_manga_data()

        # Build genre index and popularity ranking
        self._build_indexes()

        # Load the embedding model
        self._load_model()

        self.is_loaded = True
        logger.info("Search engine ready")

    def _load_manga_data(self):
        """Load full manga data from raw files."""
        page_files = sorted(RAW_DIR.glob("manga_page_*.json"))
        logger.info(f"Loading manga data from {len(page_files)} page files")

        for filepath in page_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for manga in data.get("data", []):
                        mal_id = manga.get("mal_id")
                        if mal_id:
                            self.manga_data[mal_id] = manga
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not read {filepath}: {e}")

        logger.info(f"Loaded {len(self.manga_data)} manga records")

    def _build_indexes(self):
        """Build genre index and popularity ranking."""
        genre_counts = {}
        popularity_list = []

        for mal_id, manga in self.manga_data.items():
            # Track popularity
            popularity = manga.get("popularity")
            if popularity:
                popularity_list.append((popularity, mal_id))

            # Count genres
            for genre in manga.get("genres", []):
                genre_name = genre.get("name")
                genre_id = genre.get("mal_id")
                if genre_name and genre_id:
                    if genre_name not in genre_counts:
                        genre_counts[genre_name] = {
                            "mal_id": genre_id,
                            "name": genre_name,
                            "count": 0,
                        }
                    genre_counts[genre_name]["count"] += 1

        # Sort by popularity (lower rank = more popular)
        popularity_list.sort(key=lambda x: x[0])
        self.manga_by_popularity = [mal_id for _, mal_id in popularity_list]

        # Sort genres by count
        self.genres = dict(sorted(genre_counts.items(), key=lambda x: -x[1]["count"]))

        logger.info(
            f"Indexed {len(self.genres)} genres, {len(self.manga_by_popularity)} manga by popularity"
        )

    def _load_model(self):
        """Load sentence-transformer model."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence-transformer model...")
            self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            logger.info("Model loaded")
        except ImportError:
            logger.error("sentence-transformers not installed")
            raise

    def _manga_to_result(
        self, mal_id: int, score: float = 0.0, meta: dict = None
    ) -> dict:
        """Convert manga data to result dict."""
        full_data = self.manga_data.get(mal_id, {})

        if meta is None:
            # Build meta from full_data
            meta = {
                "title": full_data.get("title", "Unknown"),
                "genres": [
                    g.get("name", "")
                    for g in full_data.get("genres", [])
                    if g.get("name")
                ],
                "themes": [
                    t.get("name", "")
                    for t in full_data.get("themes", [])
                    if t.get("name")
                ],
                "demographics": [
                    d.get("name", "")
                    for d in full_data.get("demographics", [])
                    if d.get("name")
                ],
            }

        # Extract image URL
        images = full_data.get("images", {})
        jpg_images = images.get("jpg", {})
        image_url = jpg_images.get("large_image_url") or jpg_images.get("image_url")

        return {
            "mal_id": mal_id,
            "title": meta.get("title", full_data.get("title", "Unknown")),
            "score": score,
            "genres": meta.get("genres", []),
            "themes": meta.get("themes", []),
            "demographics": meta.get("demographics", []),
            "synopsis": full_data.get("synopsis"),
            "image_url": image_url,
            "mal_score": full_data.get("score"),
            "url": full_data.get("url"),
            "popularity": full_data.get("popularity"),
            "members": full_data.get("members"),
        }

    def search(
        self, query: str, top_k: int = 10, genres: List[str] = None
    ) -> List[dict]:
        """
        Search for manga matching a natural language query.

        Args:
            query: Natural language search query
            top_k: Number of results to return
            genres: Optional list of genres to filter by

        Returns:
            List of result dicts with manga data and scores
        """
        if not self.is_loaded:
            self.load()

        # Generate query embedding
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        query_normalized = query_embedding / np.maximum(
            np.linalg.norm(query_embedding), 1e-10
        )

        # Compute cosine similarity
        similarities = np.dot(self.normalized_embeddings, query_normalized)

        # If genres specified, create a filter mask
        if genres:
            genre_set = set(g.lower() for g in genres)
            valid_indices = []
            for idx in range(len(self.mal_ids)):
                mal_id = int(self.mal_ids[idx])
                manga = self.manga_data.get(mal_id, {})
                manga_genres = set(
                    g.get("name", "").lower() for g in manga.get("genres", [])
                )
                # Intersection - has at least one matching genre
                if genre_set & manga_genres:
                    valid_indices.append(idx)

            if valid_indices:
                # Filter to only valid indices and sort by similarity
                filtered_sims = [(idx, similarities[idx]) for idx in valid_indices]
                filtered_sims.sort(key=lambda x: -x[1])
                top_indices = [idx for idx, _ in filtered_sims[:top_k]]
            else:
                top_indices = []
        else:
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results with full manga data
        results = []
        for idx in top_indices:
            mal_id = int(self.mal_ids[idx])
            meta = self.metadata[idx]
            result = self._manga_to_result(mal_id, float(similarities[idx]), meta)
            results.append(result)

        return results

    def get_top_manga(self, limit: int = 20, genres: List[str] = None) -> List[dict]:
        """
        Get top manga by popularity.

        Args:
            limit: Number of results to return
            genres: Optional list of genres to filter by

        Returns:
            List of manga sorted by popularity
        """
        if not self.is_loaded:
            self.load()

        results = []
        genre_set = set(g.lower() for g in genres) if genres else None

        for mal_id in self.manga_by_popularity:
            if len(results) >= limit:
                break

            manga = self.manga_data.get(mal_id, {})

            # Filter by genre if specified
            if genre_set:
                manga_genres = set(
                    g.get("name", "").lower() for g in manga.get("genres", [])
                )
                if not (genre_set & manga_genres):
                    continue

            result = self._manga_to_result(mal_id, score=0.0)
            results.append(result)

        return results

    def get_genres(self) -> List[dict]:
        """Get all available genres with counts."""
        if not self.is_loaded:
            self.load()

        return list(self.genres.values())

    def get_stats(self) -> dict:
        """Get statistics about the search engine."""
        if not self.is_loaded:
            self.load()

        details_dir = RAW_DIR / "manga_details"
        detail_count = (
            len(list(details_dir.glob("*.json"))) if details_dir.exists() else 0
        )

        return {
            "manga_count": len(self.mal_ids),
            "embedding_dimension": self.embeddings.shape[1],
            "has_embeddings": True,
            "detail_coverage": detail_count / len(self.mal_ids)
            if len(self.mal_ids) > 0
            else 0,
        }


# Global search engine instance
search_engine = SearchEngine()


@app.on_event("startup")
async def startup_event():
    """Load search engine on startup."""
    try:
        search_engine.load()
    except FileNotFoundError as e:
        logger.error(f"Failed to load search engine: {e}")
        logger.error("Please run the embedding service first to generate embeddings.")


@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "MangaHelp Search API",
        "version": "1.0.0",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy" if search_engine.is_loaded else "loading",
        "embeddings_loaded": search_engine.is_loaded,
    }


@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Number of results to return"),
    genres: Optional[str] = Query(None, description="Comma-separated genre filter"),
):
    """
    Search for manga using natural language queries.

    Examples:
    - "psychological thriller with unreliable narrator"
    - "wholesome slice of life with cooking"
    - "dark fantasy revenge story"

    Optionally filter by genres (comma-separated):
    - ?q=dark story&genres=Horror,Psychological
    """
    if not search_engine.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Search engine not ready. Please try again later.",
        )

    start_time = time.time()

    # Parse genres filter
    genre_list = None
    if genres:
        genre_list = [g.strip() for g in genres.split(",") if g.strip()]

    try:
        results = search_engine.search(q, top_k=limit, genres=genre_list)
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

    search_time_ms = (time.time() - start_time) * 1000

    return SearchResponse(
        query=q,
        results=[MangaResult(**r) for r in results],
        total_results=len(results),
        search_time_ms=round(search_time_ms, 2),
    )


@app.get("/top", response_model=TopMangaResponse, tags=["Discovery"])
async def get_top_manga(
    limit: int = Query(20, ge=1, le=50, description="Number of results to return"),
    genres: Optional[str] = Query(None, description="Comma-separated genre filter"),
):
    """
    Get top manga by popularity for homepage discovery.

    Optionally filter by genres:
    - ?genres=Action,Fantasy
    """
    if not search_engine.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Search engine not ready. Please try again later.",
        )

    # Parse genres filter
    genre_list = None
    if genres:
        genre_list = [g.strip() for g in genres.split(",") if g.strip()]

    try:
        results = search_engine.get_top_manga(limit=limit, genres=genre_list)
    except Exception as e:
        logger.error(f"Top manga error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get top manga")

    return TopMangaResponse(
        manga=[MangaResult(**r) for r in results],
        total=len(results),
    )


@app.get("/genres", response_model=GenresResponse, tags=["Discovery"])
async def get_genres():
    """
    Get all available genres with manga counts.

    Useful for building genre filter UI.
    """
    if not search_engine.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Search engine not ready. Please try again later.",
        )

    try:
        genres = search_engine.get_genres()
    except Exception as e:
        logger.error(f"Genres error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get genres")

    return GenresResponse(
        genres=[GenreInfo(**g) for g in genres],
        total_genres=len(genres),
    )


@app.get("/manga/{mal_id}", response_model=MangaResult, tags=["Manga"])
async def get_manga(mal_id: int):
    """Get details for a specific manga by MAL ID."""
    if not search_engine.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Search engine not ready. Please try again later.",
        )

    if mal_id not in search_engine.manga_data:
        raise HTTPException(status_code=404, detail="Manga not found")

    result = search_engine._manga_to_result(mal_id, score=0.0)
    return MangaResult(**result)


@app.get("/stats", response_model=StatsResponse, tags=["Stats"])
async def get_stats():
    """Get statistics about the search engine and data."""
    if not search_engine.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Search engine not ready. Please try again later.",
        )

    stats = search_engine.get_stats()
    return StatsResponse(**stats)


@app.get("/examples", tags=["Search"])
async def get_example_queries():
    """Get example search queries to demonstrate capabilities."""
    return {
        "examples": [
            {
                "query": "psychological thriller with mind games",
                "description": "Dark, psychological stories with mental battles",
            },
            {
                "query": "wholesome romance slice of life",
                "description": "Feel-good romantic stories",
            },
            {
                "query": "epic fantasy adventure with magic",
                "description": "Grand fantasy worlds with magical elements",
            },
            {
                "query": "revenge story with betrayal",
                "description": "Stories centered on vengeance",
            },
            {
                "query": "coming of age school drama",
                "description": "Stories about growing up in school settings",
            },
            {
                "query": "post-apocalyptic survival horror",
                "description": "Dark survival stories in ruined worlds",
            },
            {
                "query": "comedy with supernatural powers",
                "description": "Funny stories with paranormal elements",
            },
            {
                "query": "sports underdog story",
                "description": "Athletic competition with unlikely heroes",
            },
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
