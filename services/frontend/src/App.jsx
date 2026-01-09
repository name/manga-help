import React, { useState, useEffect, useCallback } from "react";

// API base URL - uses proxy in development
const API_BASE = "/api";

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchTime, setSearchTime] = useState(null);
  const [examples, setExamples] = useState([]);
  const [stats, setStats] = useState(null);
  const [genres, setGenres] = useState([]);
  const [selectedGenres, setSelectedGenres] = useState([]);
  const [topManga, setTopManga] = useState([]);
  const [loadingTop, setLoadingTop] = useState(true);
  const [showGenreFilter, setShowGenreFilter] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  // Fetch initial data on mount
  useEffect(() => {
    // Fetch examples
    fetch(`${API_BASE}/examples`)
      .then((res) => res.json())
      .then((data) => setExamples(data.examples || []))
      .catch(() => {});

    // Fetch stats
    fetch(`${API_BASE}/stats`)
      .then((res) => res.json())
      .then((data) => setStats(data))
      .catch(() => {});

    // Fetch genres
    fetch(`${API_BASE}/genres`)
      .then((res) => res.json())
      .then((data) => setGenres(data.genres || []))
      .catch(() => {});

    // Fetch top manga for homepage
    fetchTopManga();
  }, []);

  const fetchTopManga = useCallback(async (genreFilter = []) => {
    setLoadingTop(true);
    try {
      let url = `${API_BASE}/top?limit=24`;
      if (genreFilter.length > 0) {
        url += `&genres=${encodeURIComponent(genreFilter.join(","))}`;
      }
      const response = await fetch(url);
      if (response.ok) {
        const data = await response.json();
        setTopManga(data.manga || []);
      }
    } catch (err) {
      console.error("Failed to fetch top manga:", err);
    } finally {
      setLoadingTop(false);
    }
  }, []);

  // Refetch top manga when selected genres change (only if not searched)
  useEffect(() => {
    if (!hasSearched) {
      fetchTopManga(selectedGenres);
    }
  }, [selectedGenres, hasSearched, fetchTopManga]);

  const handleSearch = useCallback(
    async (searchQuery) => {
      const q = searchQuery || query;
      if (!q.trim()) return;

      setLoading(true);
      setError(null);
      setHasSearched(true);

      try {
        let url = `${API_BASE}/search?q=${encodeURIComponent(q)}&limit=24`;
        if (selectedGenres.length > 0) {
          url += `&genres=${encodeURIComponent(selectedGenres.join(","))}`;
        }

        const response = await fetch(url);

        if (!response.ok) {
          throw new Error("Search failed");
        }

        const data = await response.json();
        setResults(data.results);
        setSearchTime(data.search_time_ms);
      } catch (err) {
        setError("Failed to search. Please ensure the API is running.");
        setResults([]);
      } finally {
        setLoading(false);
      }
    },
    [query, selectedGenres],
  );

  const handleSubmit = (e) => {
    e.preventDefault();
    handleSearch();
  };

  const handleExampleClick = (exampleQuery) => {
    setQuery(exampleQuery);
    handleSearch(exampleQuery);
  };

  const handleGenreToggle = (genreName) => {
    setSelectedGenres((prev) => {
      if (prev.includes(genreName)) {
        return prev.filter((g) => g !== genreName);
      } else {
        return [...prev, genreName];
      }
    });
  };

  const clearGenres = () => {
    setSelectedGenres([]);
  };

  const handleClearSearch = () => {
    setQuery("");
    setResults([]);
    setHasSearched(false);
    setSearchTime(null);
  };

  // Determine what to show
  const showingResults = hasSearched && results.length > 0;
  const showingTop = !hasSearched && topManga.length > 0;

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <h1 className="logo" onClick={handleClearSearch} style={{ cursor: "pointer" }}>
            <span className="logo-manga">Manga</span>
            <span className="logo-help">Help</span>
          </h1>
          <p className="tagline">Semantic search for manga discovery</p>
        </div>
      </header>

      <main className="main">
        <section className="search-section">
          <form onSubmit={handleSubmit} className="search-form">
            <div className="search-input-wrapper">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Describe the manga you're looking for..."
                className="search-input"
                autoFocus
              />
              <button
                type="button"
                className={`filter-button ${selectedGenres.length > 0 ? "active" : ""}`}
                onClick={() => setShowGenreFilter(!showGenreFilter)}
                title="Filter by genre"
              >
                <svg viewBox="0 0 24 24" className="filter-icon">
                  <path d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
                </svg>
                {selectedGenres.length > 0 && <span className="filter-count">{selectedGenres.length}</span>}
              </button>
              <button type="submit" className="search-button" disabled={loading || !query.trim()}>
                {loading ? (
                  <span className="spinner"></span>
                ) : (
                  <svg viewBox="0 0 24 24" className="search-icon">
                    <path d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                )}
              </button>
            </div>
          </form>

          {/* Genre filter dropdown */}
          {showGenreFilter && (
            <div className="genre-filter">
              <div className="genre-filter-header">
                <h3>Filter by Genre</h3>
                {selectedGenres.length > 0 && (
                  <button className="clear-genres" onClick={clearGenres}>
                    Clear all
                  </button>
                )}
              </div>
              <div className="genre-grid">
                {genres.slice(0, 30).map((genre) => (
                  <button
                    key={genre.mal_id}
                    className={`genre-chip ${selectedGenres.includes(genre.name) ? "selected" : ""}`}
                    onClick={() => handleGenreToggle(genre.name)}
                  >
                    {genre.name}
                    <span className="genre-count">({genre.count})</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Example queries - only show when no search and no genre filter open */}
          {!hasSearched && !showGenreFilter && (
            <div className="examples">
              <p className="examples-label">Try searching for:</p>
              <div className="examples-grid">
                {examples.map((example, index) => (
                  <button key={index} className="example-chip" onClick={() => handleExampleClick(example.query)}>
                    {example.query}
                  </button>
                ))}
              </div>
            </div>
          )}
        </section>

        {error && <div className="error-message">{error}</div>}

        {/* Search results */}
        {showingResults && (
          <section className="results-section">
            <div className="results-header">
              <h2>
                Results for "{query}"
                {selectedGenres.length > 0 && <span className="filter-info"> in {selectedGenres.join(", ")}</span>}
              </h2>
              <div className="results-meta">
                {searchTime && (
                  <span className="search-time">
                    {results.length} results in {searchTime.toFixed(0)}ms
                  </span>
                )}
                <button className="clear-search" onClick={handleClearSearch}>
                  Clear search
                </button>
              </div>
            </div>

            <div className="results-grid">
              {results.map((manga) => (
                <MangaCard key={manga.mal_id} manga={manga} showMatchScore={true} />
              ))}
            </div>
          </section>
        )}

        {/* Top manga discovery - show on homepage */}
        {showingTop && (
          <section className="results-section">
            <div className="results-header">
              <h2>{selectedGenres.length > 0 ? `Popular ${selectedGenres.join(", ")} Manga` : "Popular Manga"}</h2>
              {loadingTop && <span className="loading-text">Loading...</span>}
            </div>

            <div className="results-grid">
              {topManga.map((manga) => (
                <MangaCard key={manga.mal_id} manga={manga} showMatchScore={false} />
              ))}
            </div>
          </section>
        )}

        {/* Empty state */}
        {hasSearched && results.length === 0 && !loading && !error && (
          <div className="empty-state">
            <p>No manga found matching your search.</p>
            <button onClick={handleClearSearch}>Browse popular manga</button>
          </div>
        )}

        {/* Stats footer */}
        {stats && (
          <footer className="stats-footer">
            <p>
              Searching {stats.manga_count.toLocaleString()} manga
              {stats.detail_coverage > 0 && (
                <span> • {(stats.detail_coverage * 100).toFixed(0)}% with character data</span>
              )}
            </p>
          </footer>
        )}
      </main>
    </div>
  );
}

function MangaCard({ manga, showMatchScore = true }) {
  const [imageError, setImageError] = useState(false);

  const scorePercent = Math.round(manga.score * 100);
  const scoreColor = scorePercent >= 40 ? "#22c55e" : scorePercent >= 25 ? "#eab308" : "#ef4444";

  return (
    <article className="manga-card">
      <div className="manga-image-wrapper">
        {manga.image_url && !imageError ? (
          <img
            src={manga.image_url}
            alt={manga.title}
            className="manga-image"
            loading="lazy"
            onError={() => setImageError(true)}
          />
        ) : (
          <div className="manga-image-placeholder">
            <span>No Image</span>
          </div>
        )}
        {showMatchScore && manga.score > 0 && (
          <div className="match-score" style={{ backgroundColor: scoreColor }}>
            {scorePercent}% match
          </div>
        )}
        {!showMatchScore && manga.popularity && <div className="popularity-badge">#{manga.popularity}</div>}
      </div>

      <div className="manga-content">
        <h3 className="manga-title">
          <a
            href={manga.url || `https://myanimelist.net/manga/${manga.mal_id}`}
            target="_blank"
            rel="noopener noreferrer"
          >
            {manga.title}
          </a>
        </h3>

        {manga.mal_score && (
          <div className="mal-score">
            <span className="star">★</span> {manga.mal_score.toFixed(2)}
            {manga.members && <span className="members"> • {(manga.members / 1000).toFixed(0)}K members</span>}
          </div>
        )}

        <div className="manga-tags">
          {manga.genres.slice(0, 3).map((genre) => (
            <span key={genre} className="tag tag-genre">
              {genre}
            </span>
          ))}
          {manga.themes.slice(0, 2).map((theme) => (
            <span key={theme} className="tag tag-theme">
              {theme}
            </span>
          ))}
        </div>

        {manga.synopsis && (
          <p className="manga-synopsis">
            {manga.synopsis.length > 200 ? manga.synopsis.slice(0, 200) + "..." : manga.synopsis}
          </p>
        )}
      </div>
    </article>
  );
}

export default App;
