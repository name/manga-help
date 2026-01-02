# Agents: Minimum Viable Architecture

This document defines the architectural boundaries and responsibilities for MangaHelp, intentionally constrained to maintain focus on semantic search as the core innovation rather than building yet another manga tracking platform.

## Core Principle
Semantic search that augments existing platforms, not replaces them.

## Service Agents

### Ingestion Agent
**Responsibility:** Batch retrieval of manga metadata from Jikan API  
**Scope:** Read-only API consumption with rate limiting compliance  
**Out of Scope:** Real-time updates, user-submitted data, content moderation

**Key Decisions:**
- Runs on schedule (daily/weekly) rather than continuously
- Stores raw JSON responses for data provenance
- Target corpus: Top 5000-10000 manga by popularity initially
- Rate limiting: 3 requests/second per Jikan documentation.

### Embedding Agent  
**Responsibility:** Convert manga metadata into searchable vector representations  
**Scope:** Generate and store embeddings using sentence-transformers  
**Out of Scope:** Custom model training, multimodal embeddings from cover art

**Key Decisions:**
- Model: sentence-transformers/all-mpnet-base-v2 (768 dimensions).
- Composite embedding strategy: concatenate synopsis + background + genre tags + themes
- Storage: PostgreSQL with pgvector extension for cosine similarity
- Batch processing after ingestion completes, not real-time

### Search API Agent
**Responsibility:** Accept natural language queries and return ranked manga results  
**Scope:** Query embedding + vector similarity search + result formatting  
**Out of Scope:** User authentication, list management, review hosting

**Key Decisions:**
- FastAPI backend for async performance
- Cosine similarity as ranking metric
- Optional personalization via imported AniList/MAL lists (phase 2)
- Response format includes mal_id, title, synopsis, score, genres, cover image

### Frontend Agent
**Responsibility:** Query interface and results display  
**Scope:** Search input, result cards with metadata, responsive design  
**Out of Scope:** User profiles, saved searches, community features

**Key Decisions:**
- Technology: Lightweight React, no framework bloat
- Hosted as static files served by API container
- Example queries visible on landing page to demonstrate capability
- Cover images proxied through backend to avoid CORS issues

## Non-Agents (Explicitly Out of Scope)

### What We're Not Building
- User authentication system (no accounts, no passwords)
- List tracking features (use AniList/MAL for this)
- Review or rating submission (read-only consumption of existing scores)
- Forum or comment system (avoid moderation politics)
- Recommendation algorithm training (use pre-trained embeddings)

### Why These Boundaries Matter
Each excluded feature represents substantial operational complexity that would distract from validating whether semantic search provides value for manga discovery. The architecture remains focused on the technical innovation while acknowledging that comprehensive tracking platforms already exist and work well.

## Success Criteria

**MVP is successful if:**
- Users can search "psychological thriller with unreliable narrator" and receive thematically relevant results
- Search completes in <500ms for queries against 5000+ manga corpus
- Results demonstrably outperform keyword matching for nuanced queries
- System runs reliably on homeserver Docker deployment
- Architecture documented sufficiently for others to replicate or extend

**MVP fails if:**
- Semantic search performs no better than simple genre filtering
- System requires constant maintenance or develops into social platform scope creep
- Embedding quality insufficient to capture meaningful distinctions between series
- Operational costs (domain + hosting) exceed value delivered to users

## Technical Constraints

- **Homeserver Deployment:** Single machine Docker Compose orchestration
- **Database:** PostgreSQL with pgvector, no separate vector database until scale demands it
- **Model Size:** Embeddings must fit in memory, ruling out models >1GB
- **API Dependencies:** Jikan only, no paid API services
- **Open Source:** MIT license, all code and architecture publicly documented

This document serves as a contract with future scope creep, when tempted to add "just one more feature," return here and verify it aligns with the core principle of augmentation rather than replacement.
