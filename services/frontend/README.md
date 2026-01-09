# MangaHelp Frontend

React-based search interface for MangaHelp semantic manga search.

## Overview

A lightweight, responsive frontend that provides a natural language search interface for discovering manga. Built with React and Vite for fast development and optimal production builds.

## Features

- **Natural Language Search**: Describe the manga you're looking for in plain English
- **Example Queries**: Click-to-search example queries to demonstrate capabilities
- **Rich Results**: Displays cover images, MAL scores, genres, themes, and synopses
- **Match Scores**: Visual indicator of how well each result matches your query
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Dark Theme**: Easy on the eyes for extended browsing

## Installation

```bash
cd services/frontend
npm install
```

## Development

### Start the Development Server

```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Prerequisites

The API must be running for search to work:

```bash
# In another terminal
cd services/api
python src/main.py
```

The Vite dev server proxies `/api` requests to `http://localhost:8000`.

### Build for Production

```bash
npm run build
```

Output will be in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## Project Structure

```
frontend/
├── index.html          # HTML entry point
├── package.json        # Dependencies and scripts
├── vite.config.js      # Vite configuration
├── public/             # Static assets
└── src/
    ├── main.jsx        # React entry point
    ├── App.jsx         # Main application component
    └── styles.css      # Global styles
```

## Configuration

### API Endpoint

In development, the Vite proxy handles API routing. For production, configure the `API_BASE` in `App.jsx`:

```javascript
// Development (uses proxy)
const API_BASE = '/api'

// Production (direct URL)
const API_BASE = 'https://your-api-domain.com'
```

### Environment Variables

Create a `.env` file for environment-specific configuration:

```
VITE_API_URL=http://localhost:8000
```

Then use in code:

```javascript
const API_BASE = import.meta.env.VITE_API_URL || '/api'
```

## Usage

1. **Search**: Type a natural language description of the manga you want
   - "psychological thriller with mind games"
   - "wholesome romance slice of life"
   - "dark fantasy revenge story"

2. **Browse Results**: View matching manga with:
   - Cover image
   - Match percentage (how well it matches your query)
   - MAL score (user rating)
   - Genres and themes
   - Synopsis excerpt

3. **Click Through**: Click manga titles to view on MyAnimeList

## Styling

The app uses CSS custom properties (variables) for theming. Key variables in `styles.css`:

```css
:root {
  --color-bg: #0f0f14;
  --color-primary: #8b5cf6;
  --color-text: #f0f0f5;
  /* ... */
}
```

## Tech Stack

- **React 18**: UI framework
- **Vite 5**: Build tool and dev server
- **CSS**: Vanilla CSS with custom properties (no framework bloat)

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Performance

- **Initial Load**: ~200KB (gzipped)
- **Search Response**: <100ms (depends on API)
- **Lighthouse Score**: 90+ (Performance, Accessibility)

## Docker Deployment

See the `Dockerfile` for containerized deployment. The frontend is served as static files by nginx.

```bash
docker build -t mangahelp-frontend .
docker run -p 80:80 mangahelp-frontend
```

## Contributing

1. Follow the existing code style
2. Keep components simple and focused
3. Avoid adding unnecessary dependencies
4. Test on mobile devices before submitting