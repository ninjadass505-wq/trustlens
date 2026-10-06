# TRUSTLENS

TRUSTLENS is an explainable scam and synthetic-media risk assessment MVP. It detects a limited set of text and URL indicators using local heuristics and reports media file properties without making an authenticity verdict.

## Requirements

- Python 3.10+
- pip

## Run locally

1. Open a terminal in `outputs/trustlens/backend`.
2. Create and activate a virtual environment:
   - Windows PowerShell: `py -m venv .venv` then `.venv\\Scripts\\Activate.ps1`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Start the server: `uvicorn main:app --reload`
5. Visit [http://127.0.0.1:8000](http://127.0.0.1:8000). API documentation is at `/docs`.

## API

- `POST /api/analyze/text` — JSON `{ "text": "..." }`
- `POST /api/analyze/url` — JSON `{ "url": "..." }`
- `POST /api/analyze/media` — multipart form with `file`
- `GET /api/health`

The static frontend is served by FastAPI. No external network reputation lookup is performed. Media analysis is limited to file type, size, and a placeholder metadata timeline; no deepfake model or audio/video synchronization detector is included. The heuristics are illustrative and should not be treated as production fraud detection. Before public deployment, configure specific CORS origins, authentication/rate limiting, privacy and retention controls, logging policy, upload malware scanning, and a validated media analysis service.
