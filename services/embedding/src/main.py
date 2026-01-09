"""
Manga Embedding Service

Converts manga metadata into searchable vector representations using
sentence-transformers for semantic search capabilities.

Model: sentence-transformers/all-mpnet-base-v2 (768 dimensions)
Strategy: Composite embeddings from synopsis + background + genres + themes + characters + moreinfo
Storage: PostgreSQL with pgvector extension (or local numpy files for testing)
"""

import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Configuration
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMENSION = 768
BATCH_SIZE = 32  # Process embeddings in batches for efficiency

# Paths - relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DETAILS_DIR = RAW_DIR / "manga_details"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
PROCESSED_DIR = DATA_DIR / "processed"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class MangaText:
    """Structured text data for a manga, ready for embedding."""

    mal_id: int
    title: str
    composite_text: str
    synopsis: str
    background: str
    genres: List[str]
    themes: List[str]
    demographics: List[str]
    characters: List[str]
    moreinfo: str


class TextComposer:
    """Composes manga metadata into text suitable for embedding."""

    def __init__(self, raw_dir: Path = RAW_DIR, details_dir: Path = DETAILS_DIR):
        self.raw_dir = raw_dir
        self.details_dir = details_dir
        self._manga_cache: Dict[int, dict] = {}
        self._details_cache: Dict[int, dict] = {}

    def load_manga_data(self) -> Dict[int, dict]:
        """Load all manga data from page files."""
        if self._manga_cache:
            return self._manga_cache

        page_files = sorted(self.raw_dir.glob("manga_page_*.json"))
        logger.info(f"Loading manga data from {len(page_files)} page files")

        for filepath in page_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for manga in data.get("data", []):
                        mal_id = manga.get("mal_id")
                        if mal_id:
                            self._manga_cache[mal_id] = manga
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not read {filepath}: {e}")

        logger.info(f"Loaded {len(self._manga_cache)} manga records")
        return self._manga_cache

    def load_details(self, mal_id: int) -> Optional[dict]:
        """Load detail data for a specific manga."""
        if mal_id in self._details_cache:
            return self._details_cache[mal_id]

        detail_file = self.details_dir / f"{mal_id}.json"
        if not detail_file.exists():
            return None

        try:
            with open(detail_file, "r", encoding="utf-8") as f:
                details = json.load(f)
                self._details_cache[mal_id] = details
                return details
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not read details for {mal_id}: {e}")
            return None

    def extract_genres(self, manga: dict) -> List[str]:
        """Extract genre names from manga data."""
        genres = manga.get("genres", [])
        return [g.get("name", "") for g in genres if g.get("name")]

    def extract_themes(self, manga: dict) -> List[str]:
        """Extract theme names from manga data."""
        themes = manga.get("themes", [])
        return [t.get("name", "") for t in themes if t.get("name")]

    def extract_demographics(self, manga: dict) -> List[str]:
        """Extract demographic names from manga data."""
        demographics = manga.get("demographics", [])
        return [d.get("name", "") for d in demographics if d.get("name")]

    def extract_characters(self, details: Optional[dict]) -> List[str]:
        """Extract character names from detail data."""
        if not details:
            return []

        characters_data = details.get("characters", {})
        if not characters_data:
            return []

        char_list = characters_data.get("data", [])
        if not char_list:
            return []

        # Get main characters first, then supporting
        main_chars = []
        supporting_chars = []

        for char_entry in char_list:
            char = char_entry.get("character", {})
            name = char.get("name", "")
            role = char_entry.get("role", "")

            if name:
                # Clean up name format (often "LastName, FirstName")
                if ", " in name:
                    parts = name.split(", ")
                    name = " ".join(reversed(parts))

                if role == "Main":
                    main_chars.append(name)
                else:
                    supporting_chars.append(name)

        # Return main characters + up to 10 supporting characters
        return main_chars + supporting_chars[:10]

    def extract_moreinfo(self, details: Optional[dict]) -> str:
        """Extract moreinfo text from detail data."""
        if not details:
            return ""

        moreinfo_data = details.get("moreinfo", {})
        if not moreinfo_data:
            return ""

        data = moreinfo_data.get("data", {})
        if not data:
            return ""

        return data.get("moreinfo", "") or ""

    def compose_text(self, mal_id: int) -> Optional[MangaText]:
        """
        Compose all text fields for a manga into a structured format.

        The composite text combines:
        - Title
        - Synopsis (main description)
        - Background (additional context)
        - Genres (categorical tags)
        - Themes (thematic elements)
        - Demographics (target audience)
        - Character names (main + supporting)
        - Moreinfo (additional notes)
        """
        manga = self._manga_cache.get(mal_id)
        if not manga:
            logger.warning(f"No manga data found for mal_id {mal_id}")
            return None

        details = self.load_details(mal_id)

        # Extract all components
        title = manga.get("title", "") or ""
        synopsis = manga.get("synopsis", "") or ""
        background = manga.get("background", "") or ""
        genres = self.extract_genres(manga)
        themes = self.extract_themes(manga)
        demographics = self.extract_demographics(manga)
        characters = self.extract_characters(details)
        moreinfo = self.extract_moreinfo(details)

        # Build composite text for embedding
        # Using a structured format that sentence-transformers can understand well
        parts = []

        # Title is important for identity
        if title:
            parts.append(f"Title: {title}")

        # Synopsis is the core description
        if synopsis:
            parts.append(f"Synopsis: {synopsis}")

        # Background provides context
        if background:
            parts.append(f"Background: {background}")

        # Categorical information
        if genres:
            parts.append(f"Genres: {', '.join(genres)}")

        if themes:
            parts.append(f"Themes: {', '.join(themes)}")

        if demographics:
            parts.append(f"Demographics: {', '.join(demographics)}")

        # Characters help identify the story
        if characters:
            parts.append(f"Characters: {', '.join(characters)}")

        # Additional info
        if moreinfo:
            parts.append(f"Additional Info: {moreinfo}")

        composite_text = "\n\n".join(parts)

        return MangaText(
            mal_id=mal_id,
            title=title,
            composite_text=composite_text,
            synopsis=synopsis,
            background=background,
            genres=genres,
            themes=themes,
            demographics=demographics,
            characters=characters,
            moreinfo=moreinfo,
        )

    def compose_all(self) -> List[MangaText]:
        """Compose text for all loaded manga."""
        self.load_manga_data()

        results = []
        for mal_id in tqdm(self._manga_cache.keys(), desc="Composing text"):
            manga_text = self.compose_text(mal_id)
            if manga_text and manga_text.composite_text.strip():
                results.append(manga_text)

        logger.info(f"Composed text for {len(results)} manga")
        return results


class EmbeddingGenerator:
    """Generates embeddings using sentence-transformers."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence-transformer model."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(
                f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}"
            )
        except ImportError:
            logger.error(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if not self.model:
            raise RuntimeError("Model not loaded")

        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding

    def generate_embeddings_batch(
        self, texts: List[str], batch_size: int = BATCH_SIZE, show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings for multiple texts in batches."""
        if not self.model:
            raise RuntimeError("Model not loaded")

        logger.info(f"Generating embeddings for {len(texts)} texts")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        return embeddings


class EmbeddingStorage:
    """Handles storage of embeddings (local files and/or PostgreSQL)."""

    def __init__(self, embeddings_dir: Path = EMBEDDINGS_DIR):
        self.embeddings_dir = embeddings_dir
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

    def save_embeddings_numpy(
        self,
        mal_ids: List[int],
        embeddings: np.ndarray,
        metadata: List[dict],
    ) -> Path:
        """
        Save embeddings to numpy files for local testing.

        Files saved:
        - embeddings.npy: The embedding vectors (N x 768)
        - mal_ids.npy: Array of mal_ids in same order
        - metadata.json: Additional metadata for each manga
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save embeddings
        embeddings_file = self.embeddings_dir / f"embeddings_{timestamp}.npy"
        np.save(embeddings_file, embeddings)
        logger.info(f"Saved embeddings to {embeddings_file}")

        # Save mal_ids
        mal_ids_file = self.embeddings_dir / f"mal_ids_{timestamp}.npy"
        np.save(mal_ids_file, np.array(mal_ids))
        logger.info(f"Saved mal_ids to {mal_ids_file}")

        # Save metadata
        metadata_file = self.embeddings_dir / f"metadata_{timestamp}.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved metadata to {metadata_file}")

        # Also save a "latest" version for easy access
        np.save(self.embeddings_dir / "embeddings_latest.npy", embeddings)
        np.save(self.embeddings_dir / "mal_ids_latest.npy", np.array(mal_ids))
        with open(
            self.embeddings_dir / "metadata_latest.json", "w", encoding="utf-8"
        ) as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return embeddings_file

    def load_latest_embeddings(self) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """Load the latest saved embeddings."""
        embeddings = np.load(self.embeddings_dir / "embeddings_latest.npy")
        mal_ids = np.load(self.embeddings_dir / "mal_ids_latest.npy")

        with open(
            self.embeddings_dir / "metadata_latest.json", "r", encoding="utf-8"
        ) as f:
            metadata = json.load(f)

        return embeddings, mal_ids, metadata


class SemanticSearch:
    """Performs semantic search over manga embeddings."""

    def __init__(
        self,
        embeddings: np.ndarray,
        mal_ids: np.ndarray,
        metadata: List[dict],
        embedding_generator: EmbeddingGenerator,
    ):
        self.embeddings = embeddings
        self.mal_ids = mal_ids
        self.metadata = metadata
        self.embedding_generator = embedding_generator

        # Normalize embeddings for cosine similarity
        self.normalized_embeddings = self._normalize(embeddings)

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-10)

    def search(self, query: str, top_k: int = 10) -> List[dict]:
        """
        Search for manga matching a natural language query.

        Args:
            query: Natural language search query
            top_k: Number of results to return

        Returns:
            List of result dicts with mal_id, title, score, and metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        query_normalized = query_embedding / np.maximum(
            np.linalg.norm(query_embedding), 1e-10
        )

        # Compute cosine similarity
        similarities = np.dot(self.normalized_embeddings, query_normalized)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            result = {
                "mal_id": int(self.mal_ids[idx]),
                "score": float(similarities[idx]),
                **self.metadata[idx],
            }
            results.append(result)

        return results


def generate_embeddings(
    raw_dir: Path = RAW_DIR,
    details_dir: Path = DETAILS_DIR,
    output_dir: Path = EMBEDDINGS_DIR,
) -> dict:
    """
    Main function to generate embeddings for all manga.

    Args:
        raw_dir: Directory containing manga page files
        details_dir: Directory containing manga detail files
        output_dir: Directory to save embeddings

    Returns:
        Summary dict with results
    """
    logger.info("Starting embedding generation")

    # Compose text
    composer = TextComposer(raw_dir, details_dir)
    manga_texts = composer.compose_all()

    if not manga_texts:
        logger.error("No manga text to embed")
        return {"status": "error", "error": "No manga data found"}

    # Generate embeddings
    generator = EmbeddingGenerator()
    texts = [mt.composite_text for mt in manga_texts]
    embeddings = generator.generate_embeddings_batch(texts)

    # Prepare metadata
    mal_ids = [mt.mal_id for mt in manga_texts]
    metadata = [
        {
            "title": mt.title,
            "genres": mt.genres,
            "themes": mt.themes,
            "demographics": mt.demographics,
        }
        for mt in manga_texts
    ]

    # Save embeddings
    storage = EmbeddingStorage(output_dir)
    embeddings_file = storage.save_embeddings_numpy(mal_ids, embeddings, metadata)

    summary = {
        "status": "completed",
        "manga_count": len(manga_texts),
        "embedding_dimension": embeddings.shape[1],
        "output_file": str(embeddings_file),
        "output_directory": str(output_dir),
    }

    logger.info("Embedding generation complete!")
    logger.info(f"Generated embeddings for {len(manga_texts)} manga")

    return summary


def interactive_search(embeddings_dir: Path = EMBEDDINGS_DIR):
    """Run an interactive search session."""
    logger.info("Loading embeddings for search...")

    storage = EmbeddingStorage(embeddings_dir)

    try:
        embeddings, mal_ids, metadata = storage.load_latest_embeddings()
    except FileNotFoundError:
        logger.error("No embeddings found. Run 'generate' first.")
        return

    generator = EmbeddingGenerator()
    search = SemanticSearch(embeddings, mal_ids, metadata, generator)

    print("\n" + "=" * 60)
    print("MangaHelp Semantic Search")
    print("=" * 60)
    print("Enter a query to search for manga, or 'quit' to exit.")
    print("Example queries:")
    print("  - psychological thriller with unreliable narrator")
    print("  - wholesome slice of life with cooking")
    print("  - dark fantasy revenge story")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break

            if not query:
                continue

            results = search.search(query, top_k=5)

            print(f"\nResults for: '{query}'")
            print("-" * 40)

            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']} (MAL ID: {result['mal_id']})")
                print(f"   Score: {result['score']:.4f}")
                if result.get("genres"):
                    print(f"   Genres: {', '.join(result['genres'])}")
                if result.get("themes"):
                    print(f"   Themes: {', '.join(result['themes'])}")

        except KeyboardInterrupt:
            print("\n")
            break

    print("Goodbye!")


def get_embedding_stats(embeddings_dir: Path = EMBEDDINGS_DIR) -> dict:
    """Get statistics about stored embeddings."""
    stats = {
        "embeddings_directory": str(embeddings_dir),
        "has_embeddings": False,
        "manga_count": 0,
        "embedding_dimension": 0,
    }

    latest_embeddings = embeddings_dir / "embeddings_latest.npy"
    if latest_embeddings.exists():
        embeddings = np.load(latest_embeddings)
        stats["has_embeddings"] = True
        stats["manga_count"] = embeddings.shape[0]
        stats["embedding_dimension"] = embeddings.shape[1]

    # List all embedding files
    embedding_files = list(embeddings_dir.glob("embeddings_*.npy"))
    stats["embedding_files"] = [
        f.name for f in embedding_files if "latest" not in f.name
    ]

    return stats


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate and search manga embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py generate          # Generate embeddings for all manga
  python main.py search            # Interactive search mode
  python main.py stats             # Show embedding statistics
  python main.py query "dark fantasy"  # Single query search
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate embeddings for all manga"
    )
    generate_parser.add_argument(
        "--output-dir",
        type=Path,
        default=EMBEDDINGS_DIR,
        help=f"Output directory for embeddings (default: {EMBEDDINGS_DIR})",
    )

    # Search command
    subparsers.add_parser("search", help="Interactive search mode")

    # Query command
    query_parser = subparsers.add_parser("query", help="Single query search")
    query_parser.add_argument("query_text", type=str, help="Search query")
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )

    # Stats command
    subparsers.add_parser("stats", help="Show embedding statistics")

    args = parser.parse_args()

    if args.command == "generate":
        summary = generate_embeddings(output_dir=args.output_dir)
        print("\n--- Embedding Generation Summary ---")
        print(json.dumps(summary, indent=2))

    elif args.command == "search":
        interactive_search()

    elif args.command == "query":
        storage = EmbeddingStorage()
        try:
            embeddings, mal_ids, metadata = storage.load_latest_embeddings()
        except FileNotFoundError:
            print("No embeddings found. Run 'generate' first.")
            return

        generator = EmbeddingGenerator()
        search = SemanticSearch(embeddings, mal_ids, metadata, generator)

        results = search.search(args.query_text, top_k=args.top_k)

        print(f"\nResults for: '{args.query_text}'")
        print("-" * 40)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']} (MAL ID: {result['mal_id']})")
            print(f"   Score: {result['score']:.4f}")
            if result.get("genres"):
                print(f"   Genres: {', '.join(result['genres'])}")
            if result.get("themes"):
                print(f"   Themes: {', '.join(result['themes'])}")

    elif args.command == "stats":
        stats = get_embedding_stats()
        print("\n--- Embedding Statistics ---")
        print(json.dumps(stats, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
