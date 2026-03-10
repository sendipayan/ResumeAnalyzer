# Resume Role Predictor API

Production-ready FastAPI service that parses a resume PDF, extracts sections, and scores fit against job roles (dataset-driven) or a single job description provided by the client.

## Features
- Resume parsing from a public PDF URL with size and network safeguards
- Section extraction for summary, experience, skills, education, projects, achievements, certificates
- Role scoring using semantic similarity + skill matching
- JD-specific scoring with custom primary/secondary skills and responsibilities
- Built-in rate limiting and optional API key auth
- CORS configuration for frontend integration

## Tech Stack
- FastAPI + Pydantic v2
- SentenceTransformers (`all-MiniLM-L6-v2`)
- scikit-learn (cosine similarity)
- RapidFuzz
- pandas / numpy
- PyMuPDF (`fitz`) for PDF text extraction
- requests for PDF fetch

Tested with Python 3.13 (see `.venv/pyvenv.cfg`).

## Project Structure
```
app/
  main.py                 # FastAPI app, endpoints, config, auth/rate limit
  dataset/job_data.csv    # Job role dataset (required)
  services/
    resume_parser.py      # PDF parsing + section extraction
    score.py              # Role scoring logic (dataset-driven)
    jdmatch.py            # JD match scoring logic (single JD)
    jobs.py               # Job link generation
```

## Quickstart
1. Create a virtualenv and install dependencies.
2. Run the API server.

Example:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install fastapi uvicorn sentence-transformers pandas numpy scikit-learn rapidfuzz pymupdf requests
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Configuration
Environment variables (with defaults):

- `APP_ENV` (default: `development`)
- `ALLOW_HTTP_RESUME_URL` (default: `true` in dev, `false` in prod)
- `ALLOW_PRIVATE_RESUME_HOSTS` (default: `false`)
- `DOWNLOAD_TIMEOUT_SECONDS` (default: `20`)
- `MAX_PDF_MB` (default: `5`)
- `RATE_LIMIT_WINDOW_SECONDS` (default: `60`)
- `RATE_LIMIT_MAX_REQUESTS` (default: `10`)
- `REQUIRE_API_KEY` (default: `false`)
- `API_KEY` (required when `REQUIRE_API_KEY=true`)
- `RESUME_DOMAIN_ALLOWLIST` (comma-separated, default: empty = allow all)
- `ALLOWED_ORIGINS` (comma-separated, defaults to common local dev hosts)

The sentence-transformer model is cached under `app/.model_cache/`.

## API

### `GET /health`
Basic health info, including model/dataset status and rate-limit settings.

### `POST /score`
Scores resume against roles from `app/dataset/job_data.csv`.

Request body:
```json
{
  "resume_url": "https://example.com/resume.pdf"
}
```

Response body:
```json
{
  "recommendations": [
    {
      "Title": "Data Scientist",
      "score": 82.3,
      "Responsibilities": ["..."],
      "primary_skill": { "coverage_score": 70, "...": "..." },
      "secondry_skill": { "coverage_score": 55, "...": "..." },
      "projects": { "semantic_score": 65, "missing": ["..."], "...": "..." },
      "experience": { "score": 74 },
      "achievment": { "final_score": 60, "...": "..." },
      "certificates": { "final_score": 40 }
    }
  ],
  "jobs": [
    { "title": "Data Scientist", "links": ["..."] }
  ],
  "resume_text": {
    "summary": "...",
    "experience": "...",
    "skills": "...",
    "education": "...",
    "projects": "...",
    "achievements": "...",
    "certificates": ["..."],
    "skills_list": ["..."],
    "experience_years": 2.5
  },
  "missing_sections": ["projects", "achievements"]
}
```

### `POST /jdmatch`
Scores resume against a single job description.

Request body:
```json
{
  "resume_url": "https://example.com/resume.pdf",
  "j_title": "Backend Engineer",
  "prim_skills": ["Python", "FastAPI"],
  "secon_skills": ["Docker", "AWS"],
  "j_resp": "Build and maintain backend services."
}
```

Response body:
```json
{
  "recommendations": [
    {
      "Title": "Backend Engineer",
      "score": 78.4,
      "Responsibilities": ["Build and maintain backend services."],
      "primary_skill": { "coverage_score": 80, "...": "..." },
      "secondry_skill": { "coverage_score": 50, "...": "..." },
      "projects": { "semantic_score": 60, "...": "..." },
      "experience": { "score": 70 },
      "achievment": { "final_score": 55, "...": "..." },
      "certificates": { "final_score": 35 }
    }
  ],
  "missing_sections": ["certificates"]
}
```

## Data Requirements
`app/dataset/job_data.csv` must exist and include at least these columns:
- `Title`
- `Responsibilities` (list or comma-separated)
- `PrimarySkills` (list or comma-separated)
- `SecondarySkills` (list or comma-separated)
- `MinYears`, `MaxYears` (numeric)

## Security Notes
- Resume URLs are validated and can be restricted via allowlist.
- Private/reserved IPs are blocked by default.
- Optional API key enforcement via `REQUIRE_API_KEY=true`.

## Development Notes
- Model loading is cached with `lru_cache`.
- PDF downloads have size and timeout limits.
- `missing_sections` is returned to indicate sections not found in the resume.
