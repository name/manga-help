# Manga Embedding Service

Converts manga metadata into searchable vector representations using sentence-transformers for semantic search capabilities.

## Overview

This service processes ingested manga data and generates dense vector embeddings that capture semantic meaning. These embeddings enable natural language search queries like "psychological thriller with unreliable narrator" to find thematically relevant manga.

## Features

- **Composite Embeddings**: Combines synopsis, background, genres, themes, characters, and moreinfo into rich representations
- **Batch Processing**: Efficiently processes large manga corpora
- **Local Storage**: Saves embeddings as numpy files for fast loading
- **Interactive Search**: Built-in search interface for testing
- **PostgreSQL Ready**: Architecture supports pgvector storage for production

## Model

- **Name**: `sentence-transformers/all-mpnet-base-v2`
- **Dimension**: 768
- **Performance**: Good balance of quality and speed for semantic similarity

## Installation

```bash
cd services/embedding
pip install -r requirements.txt
```

Note: First run will download the model (~420MB).

## Usage

### Generate Embeddings

First, ensure you have ingested manga data (see ingestion service).

```bash
python src/main.py generate
```

This will:
1. Load manga metadata from `data/raw/manga_page_*.json`
2. Load detail data from `data/raw/manga_details/*.json`
3. Compose text from all fields
4. Generate embeddings using sentence-transformers
5. Save embeddings to `data/embeddings/`

### Interactive Search

```bash
python src/main.py search
```

Enter natural language queries to search the manga corpus:

```
Query: dark fantasy with revenge and demons
Query: wholesome cooking slice of life
Query: psychological mystery with plot twists
```

### Single Query

```bash
python src/main.py query "psychological thriller with unreliable narrator"
python src/main.py query "wholesome romance" --top-k 10
```

### Check Statistics

```bash
python src/main.py stats
```

## Output

Embeddings are saved to `data/embeddings/`:

```
data/embeddings/
├── embeddings_20240108_123456.npy   # Timestamped embeddings
├── mal_ids_20240108_123456.npy      # Corresponding MAL IDs
├── metadata_20240108_123456.json    # Title, genres, themes
├── embeddings_latest.npy            # Latest embeddings (symlink)
├── mal_ids_latest.npy
└── metadata_latest.json
```

## Text Composition Strategy

Each manga's embedding is generated from a composite text that includes:

```
Title: [manga title]

Synopsis: [plot summary]

Background: [additional context, awards, publication info]

Genres: Action, Adventure, Fantasy, ...

Themes: Gore, Military, Psychological, ...

Demographics: Seinen, Shounen, ...

Characters: [Main Character 1], [Main Character 2], ...

Additional Info: [moreinfo field]
```

This approach captures both narrative content and categorical metadata.

## Performance

| Corpus Size | Embedding Time | Search Latency |
|-------------|----------------|----------------|
| 50 manga    | ~5 seconds     | <10ms          |
| 5,000 manga | ~5 minutes     | <50ms          |
| 10,000 manga| ~10 minutes    | <100ms         |

Times measured on CPU. GPU acceleration available with CUDA-enabled PyTorch.

## Search Quality

The embedding model captures semantic meaning, enabling:

- **Thematic queries**: "coming of age story with found family"
- **Mood-based queries**: "dark and depressing psychological drama"
- **Plot-based queries**: "protagonist seeks revenge against former ally"
- **Genre combinations**: "romantic comedy with supernatural elements"

## Architecture Notes

Per the project's [AGENTS.md](../../AGENTS.md):

- Model: sentence-transformers/all-mpnet-base-v2 (768 dimensions)
- Composite embedding strategy for rich representations
- Batch processing after ingestion completes
- Local numpy storage for development; PostgreSQL with pgvector for production

## Next Steps

After generating embeddings:

1. **Search API**: The embeddings can be loaded by the Search API service
2. **PostgreSQL**: For production, migrate embeddings to PostgreSQL with pgvector
3. **Incremental Updates**: Re-run generation after new ingestion runs

## Troubleshooting

### Model Download Fails

The model is downloaded from Hugging Face on first run. If it fails:

```bash
# Set Hugging Face cache directory
export HF_HOME=/path/to/cache

# Or download manually
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"
```

### Out of Memory

Reduce batch size:

```python
# In main.py, change:
BATCH_SIZE = 16  # or lower
```

### Slow on CPU

For faster processing, install PyTorch with CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
