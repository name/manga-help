"""
Manga Ingestion Service

Fetches manga metadata from the Jikan API (MyAnimeList unofficial API)
and stores raw JSON responses for later processing.

Phase 1: Fetch paginated manga lists sorted by popularity
Phase 2: Fetch per-manga details (characters, moreinfo, recommendations)

Rate Limiting: 3 requests/second per Jikan documentation
Target: Top 5000-10000 manga by popularity
Storage: Raw JSON responses in data/raw/ for data provenance
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

import requests

# Configuration
JIKAN_BASE_URL = "https://api.jikan.moe/v4"
REQUESTS_PER_SECOND = 1  # Conservative rate to avoid 429s
MIN_REQUEST_INTERVAL = 1.0 / REQUESTS_PER_SECOND  # 1 second between requests
DEFAULT_TARGET_MANGA = 5000
ITEMS_PER_PAGE = 25  # Jikan max

# Paths - check for Docker mount first, then fall back to project-relative paths
# In Docker, data is mounted at /app/data
# Locally, data is at PROJECT_ROOT/data
if Path("/app/data").exists():
    DATA_DIR = Path("/app/data")
else:
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
DETAILS_DIR = RAW_DIR / "manga_details"
PROGRESS_FILE = RAW_DIR / "ingestion_progress.json"
DETAILS_PROGRESS_FILE = RAW_DIR / "details_progress.json"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter to comply with Jikan API limits."""

    def __init__(self, requests_per_second: float = REQUESTS_PER_SECOND):
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time: Optional[float] = None

    def wait(self) -> None:
        """Wait if necessary to comply with rate limits."""
        if self.last_request_time is not None:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.3f}s")
                time.sleep(sleep_time)
        self.last_request_time = time.time()


class JikanClient:
    """Client for interacting with the Jikan API."""

    def __init__(self):
        self.base_url = JIKAN_BASE_URL
        self.rate_limiter = RateLimiter()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "MangaHelp-Ingestion/1.0",
            }
        )

    def _make_request(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        """
        Make a rate-limited request with retry logic.

        Args:
            url: Full URL to request
            params: Optional query parameters

        Returns:
            API response as dict, or None if request failed
        """
        self.rate_limiter.wait()

        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    # Resource not found - this is a valid response, not an error
                    logger.debug(f"Resource not found (404): {url}")
                    return {"data": None, "status": 404}
                elif response.status_code == 429:
                    # Rate limited - back off exponentially
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"Rate limited (429). Backing off for {delay}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                elif response.status_code >= 500:
                    # Server error - retry with backoff
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"Server error ({response.status_code}). Retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Request failed with status {response.status_code}: {response.text}"
                    )
                    return None

            except requests.exceptions.Timeout:
                delay = base_delay * (2**attempt)
                logger.warning(
                    f"Request timeout. Retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                return None

        logger.error(f"Failed to fetch {url} after {max_retries} attempts")
        return None

    def get_manga_page(self, page: int, limit: int = ITEMS_PER_PAGE) -> Optional[dict]:
        """
        Fetch a page of manga sorted by popularity.

        Args:
            page: Page number (1-indexed)
            limit: Number of items per page (max 25)

        Returns:
            API response as dict, or None if request failed
        """
        url = f"{self.base_url}/manga"
        params = {
            "page": page,
            "limit": limit,
            "order_by": "popularity",
            "sort": "asc",  # Most popular first
        }
        return self._make_request(url, params)

    def get_manga_characters(self, mal_id: int) -> Optional[dict]:
        """
        Fetch character list for a specific manga.

        Args:
            mal_id: MyAnimeList manga ID

        Returns:
            API response with character data, or None if request failed
        """
        url = f"{self.base_url}/manga/{mal_id}/characters"
        return self._make_request(url)

    def get_manga_moreinfo(self, mal_id: int) -> Optional[dict]:
        """
        Fetch additional info/notes for a specific manga.

        Args:
            mal_id: MyAnimeList manga ID

        Returns:
            API response with moreinfo data, or None if request failed
        """
        url = f"{self.base_url}/manga/{mal_id}/moreinfo"
        return self._make_request(url)

    def get_manga_recommendations(self, mal_id: int) -> Optional[dict]:
        """
        Fetch community recommendations for a specific manga.

        Args:
            mal_id: MyAnimeList manga ID

        Returns:
            API response with recommendation data, or None if request failed
        """
        url = f"{self.base_url}/manga/{mal_id}/recommendations"
        return self._make_request(url)


class IngestionProgress:
    """Tracks and persists ingestion progress for resumability."""

    def __init__(self, progress_file: Path = PROGRESS_FILE):
        self.progress_file = progress_file
        self.data = self._load()

    def _load(self) -> dict:
        """Load progress from file or return default."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load progress file: {e}")
        return {
            "last_completed_page": 0,
            "total_manga_fetched": 0,
            "last_run": None,
            "run_history": [],
        }

    def save(self) -> None:
        """Save progress to file."""
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, "w") as f:
            json.dump(self.data, f, indent=2)

    @property
    def last_completed_page(self) -> int:
        return self.data.get("last_completed_page", 0)

    @last_completed_page.setter
    def last_completed_page(self, value: int) -> None:
        self.data["last_completed_page"] = value

    @property
    def total_manga_fetched(self) -> int:
        return self.data.get("total_manga_fetched", 0)

    @total_manga_fetched.setter
    def total_manga_fetched(self, value: int) -> None:
        self.data["total_manga_fetched"] = value

    def record_run(self, manga_count: int, pages_fetched: int) -> None:
        """Record a completed ingestion run."""
        self.data["last_run"] = datetime.now().isoformat()
        self.data["run_history"].append(
            {
                "timestamp": self.data["last_run"],
                "manga_count": manga_count,
                "pages_fetched": pages_fetched,
            }
        )
        # Keep only last 10 runs
        self.data["run_history"] = self.data["run_history"][-10:]
        self.save()


class DetailIngestionProgress:
    """Tracks progress for per-manga detail fetching (Phase 2)."""

    def __init__(self, progress_file: Path = DETAILS_PROGRESS_FILE):
        self.progress_file = progress_file
        self.data = self._load()

    def _load(self) -> dict:
        """Load progress from file or return default."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load detail progress file: {e}")
        return {
            "completed_mal_ids": [],
            "total_processed": 0,
            "last_run": None,
            "include_recommendations": False,
            "run_history": [],
        }

    def save(self) -> None:
        """Save progress to file."""
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, "w") as f:
            json.dump(self.data, f, indent=2)

    @property
    def completed_mal_ids(self) -> Set[int]:
        """Get set of completed mal_ids for O(1) lookup."""
        return set(self.data.get("completed_mal_ids", []))

    def mark_completed(self, mal_id: int) -> None:
        """Mark a manga as having its details fetched."""
        if mal_id not in self.data["completed_mal_ids"]:
            self.data["completed_mal_ids"].append(mal_id)
            self.data["total_processed"] = len(self.data["completed_mal_ids"])

    @property
    def total_processed(self) -> int:
        return self.data.get("total_processed", 0)

    def record_run(self, manga_processed: int, include_recommendations: bool) -> None:
        """Record a completed detail ingestion run."""
        self.data["last_run"] = datetime.now().isoformat()
        self.data["include_recommendations"] = include_recommendations
        self.data["run_history"].append(
            {
                "timestamp": self.data["last_run"],
                "manga_processed": manga_processed,
                "include_recommendations": include_recommendations,
            }
        )
        # Keep only last 10 runs
        self.data["run_history"] = self.data["run_history"][-10:]
        self.save()

    def clear(self) -> None:
        """Clear detail progress for fresh start."""
        self.data = {
            "completed_mal_ids": [],
            "total_processed": 0,
            "last_run": None,
            "include_recommendations": False,
            "run_history": [],
        }
        self.save()


def save_raw_response(page: int, response: dict, output_dir: Path = RAW_DIR) -> Path:
    """
    Save raw API response to disk.

    Args:
        page: Page number
        response: Raw API response
        output_dir: Directory to save to

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"manga_page_{page:04d}.json"
    filepath = output_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=2)

    return filepath


def save_manga_details(
    mal_id: int,
    characters: Optional[dict],
    moreinfo: Optional[dict],
    recommendations: Optional[dict],
    output_dir: Path = DETAILS_DIR,
) -> Path:
    """
    Save per-manga detail responses to disk.

    Args:
        mal_id: MyAnimeList manga ID
        characters: Characters API response
        moreinfo: Moreinfo API response
        recommendations: Recommendations API response (can be None if skipped)
        output_dir: Directory to save to

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{mal_id}.json"

    detail_data = {
        "mal_id": mal_id,
        "fetched_at": datetime.now().isoformat(),
        "characters": characters,
        "moreinfo": moreinfo,
        "recommendations": recommendations,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(detail_data, f, ensure_ascii=False, indent=2)

    return filepath


def get_all_manga_ids(raw_dir: Path = RAW_DIR) -> List[int]:
    """
    Extract all mal_ids from fetched manga page files.

    Args:
        raw_dir: Directory containing manga_page_*.json files

    Returns:
        List of mal_ids sorted by popularity (order in files)
    """
    mal_ids = []
    page_files = sorted(raw_dir.glob("manga_page_*.json"))

    for filepath in page_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                for manga in data.get("data", []):
                    mal_id = manga.get("mal_id")
                    if mal_id and mal_id not in mal_ids:
                        mal_ids.append(mal_id)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not read {filepath}: {e}")

    return mal_ids


def ingest_manga(
    target_count: int = DEFAULT_TARGET_MANGA,
    resume: bool = True,
    output_dir: Path = RAW_DIR,
) -> dict:
    """
    Main ingestion function. Fetches manga from Jikan API.

    Args:
        target_count: Number of manga to fetch (default 5000)
        resume: Whether to resume from last progress
        output_dir: Directory to save raw responses

    Returns:
        Summary dict with ingestion results
    """
    logger.info(f"Starting manga ingestion (Phase 1). Target: {target_count} manga")
    logger.info(f"Output directory: {output_dir}")

    client = JikanClient()
    progress = IngestionProgress()

    # Calculate pages needed
    pages_needed = (target_count + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE
    start_page = progress.last_completed_page + 1 if resume else 1

    if resume and start_page > 1:
        logger.info(
            f"Resuming from page {start_page} (previously fetched {progress.total_manga_fetched} manga)"
        )
    else:
        # Reset progress if not resuming
        progress.last_completed_page = 0
        progress.total_manga_fetched = 0

    manga_fetched = progress.total_manga_fetched
    pages_fetched_this_run = 0
    errors = []

    for page in range(start_page, pages_needed + 1):
        logger.info(
            f"Fetching page {page}/{pages_needed} ({manga_fetched}/{target_count} manga)"
        )

        response = client.get_manga_page(page)

        if response is None:
            error_msg = f"Failed to fetch page {page}"
            logger.error(error_msg)
            errors.append(error_msg)
            # Continue to next page instead of stopping entirely
            continue

        # Save raw response
        filepath = save_raw_response(page, response, output_dir)
        logger.debug(f"Saved to {filepath}")

        # Update progress
        page_manga_count = len(response.get("data", []))
        manga_fetched += page_manga_count
        pages_fetched_this_run += 1

        progress.last_completed_page = page
        progress.total_manga_fetched = manga_fetched
        progress.save()

        # Check if we've reached target
        if manga_fetched >= target_count:
            logger.info(f"Reached target of {target_count} manga")
            break

        # Check if no more pages
        pagination = response.get("pagination", {})
        if not pagination.get("has_next_page", False):
            logger.info("No more pages available")
            break

    # Record the run
    progress.record_run(manga_fetched, pages_fetched_this_run)

    summary = {
        "status": "completed" if not errors else "completed_with_errors",
        "total_manga_fetched": manga_fetched,
        "pages_fetched_this_run": pages_fetched_this_run,
        "total_pages_completed": progress.last_completed_page,
        "errors": errors,
        "output_directory": str(output_dir),
    }

    logger.info("Ingestion complete!")
    logger.info(f"Total manga fetched: {manga_fetched}")
    logger.info(f"Pages fetched this run: {pages_fetched_this_run}")
    if errors:
        logger.warning(f"Errors encountered: {len(errors)}")

    return summary


def ingest_manga_details(
    include_recommendations: bool = False,
    resume: bool = True,
    limit: Optional[int] = None,
    raw_dir: Path = RAW_DIR,
    output_dir: Path = DETAILS_DIR,
) -> dict:
    """
    Phase 2: Fetch per-manga details (characters, moreinfo, recommendations).

    Args:
        include_recommendations: Whether to fetch recommendations (adds ~33% more requests)
        resume: Whether to resume from last progress
        limit: Optional limit on number of manga to process (for testing)
        raw_dir: Directory containing manga page files
        output_dir: Directory to save detail responses

    Returns:
        Summary dict with ingestion results
    """
    logger.info("Starting manga detail ingestion (Phase 2)")
    logger.info(f"Include recommendations: {include_recommendations}")
    logger.info(f"Output directory: {output_dir}")

    # Get all manga IDs from Phase 1 data
    all_mal_ids = get_all_manga_ids(raw_dir)
    if not all_mal_ids:
        logger.error("No manga IDs found. Run 'ingest' first to fetch manga list.")
        return {
            "status": "error",
            "error": "No manga IDs found. Run Phase 1 first.",
        }

    logger.info(f"Found {len(all_mal_ids)} manga IDs to process")

    # Apply limit if specified
    if limit:
        all_mal_ids = all_mal_ids[:limit]
        logger.info(f"Limited to first {limit} manga")

    # Load progress
    progress = DetailIngestionProgress()
    completed_ids = progress.completed_mal_ids if resume else set()

    if resume and completed_ids:
        logger.info(f"Resuming: {len(completed_ids)} manga already processed")
    elif not resume:
        progress.clear()
        completed_ids = set()

    # Filter to only unprocessed manga
    pending_ids = [mid for mid in all_mal_ids if mid not in completed_ids]
    logger.info(f"Manga to process: {len(pending_ids)}")

    if not pending_ids:
        logger.info("All manga details already fetched!")
        return {
            "status": "completed",
            "total_processed": len(completed_ids),
            "processed_this_run": 0,
            "output_directory": str(output_dir),
        }

    # Estimate time
    requests_per_manga = 3 if include_recommendations else 2
    total_requests = len(pending_ids) * requests_per_manga
    estimated_seconds = total_requests / REQUESTS_PER_SECOND
    estimated_minutes = estimated_seconds / 60
    logger.info(
        f"Estimated time: {estimated_minutes:.1f} minutes ({total_requests} requests)"
    )

    client = JikanClient()
    processed_this_run = 0
    errors = []

    for i, mal_id in enumerate(pending_ids, 1):
        logger.info(f"Processing manga {mal_id} ({i}/{len(pending_ids)})")

        # Fetch characters
        characters = client.get_manga_characters(mal_id)
        if characters is None:
            errors.append(f"Failed to fetch characters for {mal_id}")

        # Fetch moreinfo
        moreinfo = client.get_manga_moreinfo(mal_id)
        if moreinfo is None:
            errors.append(f"Failed to fetch moreinfo for {mal_id}")

        # Fetch recommendations (optional)
        recommendations = None
        if include_recommendations:
            recommendations = client.get_manga_recommendations(mal_id)
            if recommendations is None:
                errors.append(f"Failed to fetch recommendations for {mal_id}")

        # Save details (even if some requests failed, save what we got)
        save_manga_details(mal_id, characters, moreinfo, recommendations, output_dir)

        # Update progress
        progress.mark_completed(mal_id)
        progress.save()
        processed_this_run += 1

        # Log progress every 50 manga
        if processed_this_run % 50 == 0:
            logger.info(
                f"Progress: {processed_this_run}/{len(pending_ids)} manga processed"
            )

    # Record the run
    progress.record_run(processed_this_run, include_recommendations)

    summary = {
        "status": "completed" if not errors else "completed_with_errors",
        "total_processed": progress.total_processed,
        "processed_this_run": processed_this_run,
        "include_recommendations": include_recommendations,
        "errors": errors[:20] if errors else [],  # Limit error list
        "total_errors": len(errors),
        "output_directory": str(output_dir),
    }

    logger.info("Detail ingestion complete!")
    logger.info(f"Processed this run: {processed_this_run}")
    logger.info(f"Total processed: {progress.total_processed}")
    if errors:
        logger.warning(f"Errors encountered: {len(errors)}")

    return summary


def get_ingestion_stats(
    output_dir: Path = RAW_DIR, details_dir: Path = DETAILS_DIR
) -> dict:
    """Get statistics about current ingestion data."""
    progress = IngestionProgress()
    detail_progress = DetailIngestionProgress()

    # Count raw files
    raw_files = list(output_dir.glob("manga_page_*.json"))

    # Count total manga in files
    total_manga = 0
    mal_ids = set()
    for filepath in raw_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                manga_list = data.get("data", [])
                total_manga += len(manga_list)
                for manga in manga_list:
                    mal_ids.add(manga.get("mal_id"))
        except (json.JSONDecodeError, IOError):
            pass

    # Count detail files
    detail_files = list(details_dir.glob("*.json")) if details_dir.exists() else []

    return {
        "phase1": {
            "raw_files_count": len(raw_files),
            "total_manga_records": total_manga,
            "unique_manga_ids": len(mal_ids),
            "last_completed_page": progress.last_completed_page,
            "last_run": progress.data.get("last_run"),
        },
        "phase2": {
            "detail_files_count": len(detail_files),
            "manga_with_details": detail_progress.total_processed,
            "last_run": detail_progress.data.get("last_run"),
            "include_recommendations": detail_progress.data.get(
                "include_recommendations"
            ),
        },
        "output_directory": str(output_dir),
        "details_directory": str(details_dir),
    }


def clear_progress(phase: str = "all") -> None:
    """
    Clear ingestion progress (for fresh start).

    Args:
        phase: Which phase to clear - "1", "2", or "all"
    """
    if phase in ("1", "all"):
        progress = IngestionProgress()
        progress.data = {
            "last_completed_page": 0,
            "total_manga_fetched": 0,
            "last_run": None,
            "run_history": [],
        }
        progress.save()
        logger.info("Phase 1 progress cleared")

    if phase in ("2", "all"):
        detail_progress = DetailIngestionProgress()
        detail_progress.clear()
        logger.info("Phase 2 progress cleared")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest manga metadata from Jikan API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 1: Fetch manga list
  python main.py ingest                    # Fetch 5000 manga (default)
  python main.py ingest --target 10000     # Fetch 10000 manga
  python main.py ingest --no-resume        # Start fresh, don't resume

  # Phase 2: Fetch per-manga details
  python main.py ingest-details            # Fetch characters + moreinfo
  python main.py ingest-details --with-recommendations  # Include recommendations
  python main.py ingest-details --limit 100             # Process only 100 manga (testing)

  # Utilities
  python main.py stats                     # Show current ingestion stats
  python main.py clear                     # Clear all progress
  python main.py clear --phase 2           # Clear only Phase 2 progress
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command (Phase 1)
    ingest_parser = subparsers.add_parser(
        "ingest", help="Phase 1: Fetch manga list from Jikan API"
    )
    ingest_parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET_MANGA,
        help=f"Target number of manga to fetch (default: {DEFAULT_TARGET_MANGA})",
    )
    ingest_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from last progress",
    )
    ingest_parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Output directory for raw JSON files (default: {RAW_DIR})",
    )

    # Ingest-details command (Phase 2)
    details_parser = subparsers.add_parser(
        "ingest-details",
        help="Phase 2: Fetch per-manga details (characters, moreinfo, recommendations)",
    )
    details_parser.add_argument(
        "--with-recommendations",
        action="store_true",
        help="Include recommendations (adds ~33%% more requests)",
    )
    details_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from last progress",
    )
    details_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of manga to process (for testing)",
    )
    details_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DETAILS_DIR,
        help=f"Output directory for detail JSON files (default: {DETAILS_DIR})",
    )

    # Stats command
    subparsers.add_parser("stats", help="Show ingestion statistics")

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear ingestion progress")
    clear_parser.add_argument(
        "--phase",
        choices=["1", "2", "all"],
        default="all",
        help="Which phase progress to clear (default: all)",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        summary = ingest_manga(
            target_count=args.target,
            resume=not args.no_resume,
            output_dir=args.output_dir,
        )
        print("\n--- Phase 1 Ingestion Summary ---")
        print(json.dumps(summary, indent=2))

    elif args.command == "ingest-details":
        summary = ingest_manga_details(
            include_recommendations=args.with_recommendations,
            resume=not args.no_resume,
            limit=args.limit,
            output_dir=args.output_dir,
        )
        print("\n--- Phase 2 Detail Ingestion Summary ---")
        print(json.dumps(summary, indent=2))

    elif args.command == "stats":
        stats = get_ingestion_stats()
        print("\n--- Ingestion Statistics ---")
        print(json.dumps(stats, indent=2))

    elif args.command == "clear":
        clear_progress(args.phase)
        print(f"Progress cleared (phase: {args.phase})")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
