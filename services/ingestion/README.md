# Manga Ingestion Service

Batch retrieval of manga metadata from the [Jikan API](https://jikan.moe/) (unofficial MyAnimeList API).

## Overview

This service fetches manga metadata in two phases:

1. **Phase 1**: Fetch paginated manga lists sorted by popularity
2. **Phase 2**: Fetch per-manga details (characters, moreinfo, recommendations)

Raw JSON responses are stored for data provenance, and the process is designed to build a corpus of the top 5,000-10,000 manga for semantic search.

## Features

- **Rate Limiting**: Complies with Jikan's 3 requests/second limit with automatic backoff on 429 errors
- **Resumable**: Tracks progress for both phases and can resume from where it left off
- **Data Provenance**: Stores raw API responses for reproducibility
- **Error Handling**: Retries failed requests with exponential backoff
- **Flexible Detail Fetching**: Optional inclusion of recommendations data

## Installation

```bash
cd services/ingestion
pip install -r requirements.txt
```

## Usage

### Phase 1: Fetch Manga List

#### Fetch manga (default: 5,000)

```bash
python src/main.py ingest
```

#### Fetch a specific number of manga

```bash
python src/main.py ingest --target 10000
```

#### Start fresh (don't resume from previous progress)

```bash
python src/main.py ingest --no-resume
```

### Phase 2: Fetch Per-Manga Details

After Phase 1 completes, fetch additional details for each manga:

#### Fetch characters and moreinfo

```bash
python src/main.py ingest-details
```

#### Include recommendations (adds ~33% more requests)

```bash
python src/main.py ingest-details --with-recommendations
```

#### Test with a limited number of manga

```bash
python src/main.py ingest-details --limit 100
```

#### Start fresh (don't resume)

```bash
python src/main.py ingest-details --no-resume
```

### Utilities

#### Check current ingestion statistics

```bash
python src/main.py stats
```

#### Clear progress to start over

```bash
# Clear all progress
python src/main.py clear

# Clear only Phase 1 progress
python src/main.py clear --phase 1

# Clear only Phase 2 progress
python src/main.py clear --phase 2
```

## Output

### Phase 1 Output

Raw JSON responses are saved to `data/raw/` with the naming pattern:

```
data/raw/
├── manga_page_0001.json
├── manga_page_0002.json
├── ...
└── ingestion_progress.json
```

Each page file contains:
- `pagination`: Metadata about pagination state
- `data`: Array of manga objects with full metadata

### Phase 2 Output

Per-manga detail responses are saved to `data/raw/manga_details/`:

```
data/raw/manga_details/
├── 1.json           # mal_id 1
├── 2.json           # mal_id 2
├── ...
└── details_progress.json
```

Each detail file contains:
- `mal_id`: MyAnimeList ID
- `fetched_at`: Timestamp of fetch
- `characters`: Character list with names, roles, and images
- `moreinfo`: Additional free-text notes
- `recommendations`: Community recommendations (if fetched)

### Manga Object Fields (Phase 1)

Each manga record includes:
- `mal_id`: MyAnimeList ID (unique identifier)
- `title`, `title_english`, `title_japanese`: Title variants
- `synopsis`: Plot summary
- `background`: Additional context/history
- `genres`: Array of genre objects
- `themes`: Array of theme objects
- `demographics`: Target audience
- `score`: User rating
- `popularity`: Popularity rank
- `status`: Publishing status
- `authors`: Author information
- `images`: Cover image URLs

### Character Object Fields (Phase 2)

Each character record includes:
- `character.mal_id`: Character ID
- `character.name`: Character name
- `character.images`: Character image URLs
- `role`: Role in the manga (Main, Supporting)

## Time Estimates

Rate limit: 1 request per second (conservative to avoid 429 errors)

### Phase 1

- **5,000 manga**: ~200 requests ≈ 3.3 minutes
- **10,000 manga**: ~400 requests ≈ 6.7 minutes

### Phase 2

For each manga, 2-3 requests are made depending on options:

| Manga Count | Without Recommendations | With Recommendations |
|-------------|------------------------|---------------------|
| 5,000       | ~10,000 reqs ≈ 167 min (~2.8 hrs) | ~15,000 reqs ≈ 250 min (~4.2 hrs) |
| 10,000      | ~20,000 reqs ≈ 333 min (~5.6 hrs) | ~30,000 reqs ≈ 500 min (~8.3 hrs) |

## Rate Limiting

The Jikan API has the following limits:
- **3 requests per second** (documented)
- Automatic 429 response when exceeded

This service implements:
- **1 request per second** (conservative to avoid rate limit errors)
- Exponential backoff on rate limit errors (429)
- Retry logic for server errors (5xx) and timeouts

Note: While Jikan documents 3 req/sec, in practice a 1 req/sec rate is more reliable
and avoids frequent backoffs that slow down overall ingestion time.

## Architecture Notes

Per the project's [AGENTS.md](../../AGENTS.md):

- Runs on schedule (daily/weekly) rather than continuously
- Stores raw JSON responses for data provenance
- Target corpus: Top 5,000-10,000 manga by popularity
- Read-only API consumption (no writes to external services)

## Workflow

A typical ingestion workflow:

```bash
# 1. Fetch manga list (Phase 1)
python src/main.py ingest --target 5000

# 2. Check progress
python src/main.py stats

# 3. Fetch details for all manga (Phase 2)
python src/main.py ingest-details

# 4. Verify completion
python src/main.py stats
```

For testing or development:

```bash
# Fetch a small batch
python src/main.py ingest --target 100

# Fetch details for just 10 manga
python src/main.py ingest-details --limit 10
```
