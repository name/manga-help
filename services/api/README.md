# MangaHelp Search API

FastAPI backend for semantic manga search using pre-computed embeddings.

## Overview

This service provides a REST API for searching manga using natural language queries. It loads pre-computed embeddings and performs cosine similarity search to find semantically relevant manga.

## Features

- **Semantic Search**: Natural language queries like "psychological thriller with unreliable narrator"
- **Fast Response**: Sub-100ms search latency with pre-loaded embeddings
- **Rich Results**: Returns title, synopsis, genres, themes, cover images, and MAL scores
- **CORS Enabled**: Ready for frontend integration

## Installation

```bash
cd services/api
pip install -r requirements.txt
```

## Usage

### Start the Development Server

```bash
# From the api directory
python src/main.py

# Or with uvicorn directly
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

### Search

```
GET /search?q={query}&limit={limit}
```

Search for manga using natural language.

**Parameters:**
- `q` (required): Search query (1-500 characters)
- `limit` (optional): Number of results (1-50, default: 10)

**Example:**
```bash
curl "http://localhost:8000/search?q=dark%20fantasy%20revenge&limit=5"
```

**Response:**
```json
{
  "query": "dark fantasy revenge",
  "results": [
    {
      "mal_id": 2,
      "title": "Berserk",
      "score": 0.4523,
      "genres": ["Action", "Adventure", "Drama", "Fantasy", "Horror"],
      "themes": ["Gore", "Military", "Psychological"],
      "demographics": ["Seinen"],
      "synopsis": "Guts, a former mercenary...",
      "image_url": "https://cdn.myanimelist.net/images/manga/1/157897l.jpg",
      "mal_score": 9.47,
      "url": "https://myanimelist.net/manga/2/Berserk"
    }
  ],
  "total_results": 5,
  "search_time_ms": 45.23
}
```

### Get Manga by ID

```
GET /manga/{mal_id}
```

Get details for a specific manga.

**Example:**
```bash
curl "http://localhost:8000/manga/2"
```

### Statistics

```
GET /stats
```

Get statistics about the search engine.

**Response:**
```json
{
  "manga_count": 9962,
  "embedding_dimension": 768,
  "has_embeddings": true,
  "detail_coverage": 0.178
}
```

### Example Queries

```
GET /examples
```

Get example search queries to demonstrate capabilities.

### Health Check

```
GET /health
```

Check if the service is ready.

## Configuration

Environment variables (optional):

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |

## Architecture

```
Request Flow:
1. Client sends query to /search
2. API encodes query using sentence-transformer model
3. Computes cosine similarity against pre-loaded embeddings
4. Returns top-k results with full manga metadata
```

## Dependencies

- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **sentence-transformers**: Query embedding
- **NumPy**: Vector operations

## Prerequisites

Before starting the API, ensure:

1. Embeddings have been generated:
   ```bash
   python services/embedding/src/main.py generate
   ```

2. Embedding files exist at:
   - `data/embeddings/embeddings_latest.npy`
   - `data/embeddings/mal_ids_latest.npy`
   - `data/embeddings/metadata_latest.json`

## Performance

| Metric | Value |
|--------|-------|
| Startup time | ~10 seconds (model loading) |
| Search latency | <100ms |
| Memory usage | ~2GB (model + embeddings) |

## Production Deployment

For production, use Gunicorn with Uvicorn workers:

```bash
gunicorn src.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

See the Docker configuration for containerized deployment.