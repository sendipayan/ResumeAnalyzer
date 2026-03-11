import ipaddress
import logging
import os
import socket
import ast
import time
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import AnyHttpUrl, BaseModel, Field
from sentence_transformers import SentenceTransformer

from app.services.resume_parser import CVPipeline
from app.services.score import RolePredicter
from app.services.jdmatch import JDMatch
from app.services.jobs import JobRedirectBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
JOB_DATA_PATH = BASE_DIR / "dataset" / "job_data.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_CACHE_DIR = BASE_DIR / ".model_cache"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
ALLOW_HTTP_RESUME_URL = os.getenv(
    "ALLOW_HTTP_RESUME_URL", "true" if APP_ENV != "production" else "false"
).strip().lower() == "true"
ALLOW_PRIVATE_RESUME_HOSTS = os.getenv("ALLOW_PRIVATE_RESUME_HOSTS", "false").strip().lower() == "true"
DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "20"))
MAX_PDF_MB = int(os.getenv("MAX_PDF_MB", "5"))
MAX_PDF_BYTES = MAX_PDF_MB * 1024 * 1024
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "10"))
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").strip().lower() == "true"
API_KEY = os.getenv("API_KEY")
RESUME_DOMAIN_ALLOWLIST = [
    domain.strip().lower()
    for domain in os.getenv("RESUME_DOMAIN_ALLOWLIST", "").split(",")
    if domain.strip()
]
DEFAULT_DEV_ORIGINS = "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173"
_origins_raw = os.getenv(
    "ALLOWED_ORIGINS", DEFAULT_DEV_ORIGINS if APP_ENV != "production" else ""
)
ALLOWED_ORIGINS = [origin.strip() for origin in _origins_raw.split(",") if origin.strip()]

if MAX_PDF_MB <= 0:
    MAX_PDF_MB = 5
    MAX_PDF_BYTES = MAX_PDF_MB * 1024 * 1024

_request_log: dict[str, deque[float]] = defaultdict(deque)
_request_log_lock = Lock()
extractor = CVPipeline()



@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    logger.info("Loading SentenceTransformer model: %s", MODEL_NAME)
    return SentenceTransformer(MODEL_NAME, cache_folder=str(MODEL_CACHE_DIR))


@lru_cache(maxsize=1)
def get_job_df() -> pd.DataFrame:
    if not JOB_DATA_PATH.exists():
        raise FileNotFoundError(f"Job dataset not found at: {JOB_DATA_PATH}")



    data = pd.read_csv(JOB_DATA_PATH)

    def parse_list_column(x):
        if pd.isna(x):
            return []

        # Already a list (unlikely but safe)
        if isinstance(x, list):
            return x

        if isinstance(x, str):
            x = x.strip()

            # If stored as stringified list: "['React', 'Node']"
            if x.startswith("[") and x.endswith("]"):
                try:
                    return ast.literal_eval(x)
                except Exception:
                    pass

            # Fallback: comma separated
            return [s.strip() for s in x.split(",")]

        return []

    list_columns = [
    "Skills",
    "CleanedSkills",
    "PrimarySkills",
    "SecondarySkills",
    "Responsibilities",
    "Keywords",
    ]

    for col in list_columns:
        if col in data.columns:
            data[col] = data[col].apply(parse_list_column)

    return data


def to_builtin(value):
    if isinstance(value, dict):  
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _missing_resume_sections(resume: dict[str, Any]) -> list[str]:
    sections = {
        "summary": resume.get("summary"),
        "experience": resume.get("experience"),
        "skills": resume.get("skills"),
        "education": resume.get("education"),
        "projects": resume.get("projects"),
        "achievements": resume.get("achievements"),
        "certificates": resume.get("certificates"),
    }

    missing: list[str] = []
    for name, value in sections.items():
        if value is None:
            missing.append(name)
            continue
        if isinstance(value, str):
            if not value.strip():
                missing.append(name)
            continue
        if isinstance(value, (list, tuple, set, dict)):
            if len(value) == 0:
                missing.append(name)
            continue
        if not value:
            missing.append(name)

    return missing


def _is_blocked_ip(ip_text: str) -> bool:
    ip_obj = ipaddress.ip_address(ip_text)
    return (
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_reserved
        or ip_obj.is_multicast
        or ip_obj.is_unspecified
    )


def _host_matches_allowlist(host: str) -> bool:
    if not RESUME_DOMAIN_ALLOWLIST:
        return True
    return any(host == domain or host.endswith(f".{domain}") for domain in RESUME_DOMAIN_ALLOWLIST)


def validate_resume_url(resume_url: AnyHttpUrl) -> str:
    url_str = str(resume_url)
    parsed = urlparse(url_str)

    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="resume_url must use http or https")
    if parsed.scheme == "http" and not ALLOW_HTTP_RESUME_URL:
        raise HTTPException(status_code=400, detail="resume_url must use https")
    if parsed.username or parsed.password:
        raise HTTPException(status_code=400, detail="resume_url must not include credentials")

    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise HTTPException(status_code=400, detail="resume_url host is missing")
    if not _host_matches_allowlist(host):
        raise HTTPException(status_code=400, detail="resume_url host is not in the allowlist")

    if not ALLOW_PRIVATE_RESUME_HOSTS:
        try:
            addr_info = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        except socket.gaierror as exc:
            raise HTTPException(status_code=400, detail="resume_url host could not be resolved") from exc

        for entry in addr_info:
            ip_text = entry[4][0]
            if _is_blocked_ip(ip_text):
                raise HTTPException(
                    status_code=400,
                    detail="resume_url host resolves to a private or restricted network",
                )

    return url_str


app = FastAPI(
    title="Resume Role Predictor API",
    description="Parse resume from URL and return ranked role recommendations.",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def auth_and_rate_limit_middleware(request: Request, call_next):
    if request.url.path in {"/score", "/jdmatch", "/ats"}:
        if REQUIRE_API_KEY:
            if not API_KEY:
                logger.error("REQUIRE_API_KEY is enabled but API_KEY is not set")
                return JSONResponse(status_code=503, content={"detail": "Service authentication is not configured"})
            provided_key = request.headers.get("x-api-key")
            if not provided_key or provided_key != API_KEY:
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        cutoff = now - RATE_LIMIT_WINDOW_SECONDS

        with _request_log_lock:
            bucket = _request_log[client_ip]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Please retry later."},
                )
            bucket.append(now)

    return await call_next(request)


class ResumeRequest(BaseModel):
    resume_url: AnyHttpUrl = Field(..., description="Public URL of the resume PDF")


class JDMatchRequest(ResumeRequest):
    j_title: str = Field(..., description="Job title")
    prim_skills: list[str] = Field(default_factory=list, description="Primary skills")
    secon_skills: list[str] = Field(default_factory=list, description="Secondary skills")
    j_resp: str = Field(..., description="Job responsibilities text")
    resume_url: AnyHttpUrl = Field(..., description="Public URL of the resume PDF")


class ScoreResponse(BaseModel):
    recommendations: list[dict[str, Any]]
    jobs: list[dict[str, Any]]
    resume_text: dict[str, Any]
    missing_sections: list[str] = Field(default_factory=list, description="Resume sections not found")


class JDMatchResponse(BaseModel):
    recommendations: list[dict[str, Any]]
    missing_sections: list[str] = Field(default_factory=list, description="Resume sections not found")


class ATSResponse(BaseModel):
    text: str
    ATS_score: float
    section_score: float
    contact_score: float
    formating_score: float
    issues: list[str] = Field(default_factory=list)


@app.get("/health")
def health_check() -> dict:
    model_loaded = True
    dataset_loaded = True
    errors: list[str] = []

    try:
        get_model()
    except Exception as exc:
        model_loaded = False
        errors.append(f"model_error: {exc}")

    try:
        get_job_df()
    except Exception as exc:
        dataset_loaded = False
        errors.append(f"dataset_error: {exc}")

    status = "healthy" if model_loaded and dataset_loaded else "degraded"
    return {
        "status": status,
        "model": MODEL_NAME,
        "model_loaded": model_loaded,
        "job_data_loaded": dataset_loaded,
        "api_key_required": REQUIRE_API_KEY,
        "rate_limit": {
            "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
            "max_requests": RATE_LIMIT_MAX_REQUESTS,
        },
        "errors": errors,
    }


@app.get("/")
def root() -> dict:
    return {
        "message": "Resume Role Predictor API",
        "endpoints": {
            "/health": "Health check",
            "/score": "Score resume against roles (POST)",
            "/jdmatch": "Score resume against a frontend-provided JD (POST)",
            "/ats": "ATS-friendly score for resume (POST)",
            "/docs": "Swagger UI",
        },
    }


@app.post("/score", response_model=ScoreResponse)
def score_resume(payload: ResumeRequest) -> ScoreResponse:
    safe_resume_url = validate_resume_url(payload.resume_url)
    job_links=[]
    try:
        model = get_model()
    except Exception as exc:
        logger.exception("Model load failed")
        raise HTTPException(status_code=503, detail="Model not available") from exc

    try:
        job_df = get_job_df()
    except Exception as exc:
        logger.exception("Dataset load failed")
        raise HTTPException(status_code=503, detail="Dataset not available") from exc

    try:
        resume = extractor.run(
            pdf_url=safe_resume_url,
            download_timeout=DOWNLOAD_TIMEOUT_SECONDS,
            max_pdf_bytes=MAX_PDF_BYTES,
        )
    except ValueError as exc:
        logger.warning("Resume parsing validation failed for URL: %s", payload.resume_url)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Resume parsing failed for URL: %s", payload.resume_url)
        raise HTTPException(status_code=400, detail="Failed to parse resume PDF") from exc
    
    missing_sections = _missing_resume_sections(resume)


    try:
        scorer = RolePredicter(
            resume_skills=resume["skills_list"],
            job_df=job_df,
            model=model,
            project_text=resume["projects"],
            exp_text=resume["experience"],
            achiev_text=resume["achievements"],
            cert_list=resume["certificates"],
        )
        
        result = scorer.final_score(exp_year=resume["experience_years"])
    except Exception as exc:
        logger.exception("Resume scoring failed")
        raise HTTPException(status_code=500, detail="Failed to score resume") from exc
    
    try:
        for r in result:
            links={}
            jobs=JobRedirectBuilder(role=r["Title"],
                                skills=r["primary_skill"]["matched"],
                                experience_years=resume["experience_years"])
            links["title"]=r["Title"]
            links["links"]=jobs.generate_links()
            job_links.append(links)
        
    except Exception as exc:
        logger.exception("Job fetching failed for the resume URL: %s", payload.resume_url)
        raise HTTPException(status_code=500, detail="Failed to find matching jobs") from exc

    recommendations = to_builtin(result)
    for recommendation in recommendations:
        if "score" in recommendation:
            recommendation["score"] = min(recommendation["score"], 100)

    return ScoreResponse(
        recommendations=recommendations,
        jobs=job_links,
        resume_text=resume,
        missing_sections=missing_sections,
    )


@app.post("/jdmatch", response_model=JDMatchResponse)
def score_resume_with_jd(payload: JDMatchRequest) -> JDMatchResponse:
    safe_resume_url = validate_resume_url(payload.resume_url)

    try:
        model = get_model()
    except Exception as exc:
        logger.exception("Model load failed")
        raise HTTPException(status_code=503, detail="Model not available") from exc

    try:
        resume = extractor.run(
            pdf_url=safe_resume_url,
            download_timeout=DOWNLOAD_TIMEOUT_SECONDS,
            max_pdf_bytes=MAX_PDF_BYTES,
        )
    except ValueError as exc:
        logger.warning("Resume parsing validation failed for URL: %s", payload.resume_url)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Resume parsing failed for URL: %s", payload.resume_url)
        raise HTTPException(status_code=400, detail="Failed to parse resume PDF") from exc
    
    missing_sections = _missing_resume_sections(resume)

    try:
        jd_matcher = JDMatch(
            resume_skills=resume["skills_list"],
            model=model,
            project_text=resume["projects"],
            exp_text=resume["experience"],
            achiev_text=resume["achievements"],
            cert_list=resume["certificates"],
            j_title=payload.j_title,
            prim_skills=payload.prim_skills,
            secod_skills=payload.secon_skills,
            j_resp=[payload.j_resp],
        )
        result = jd_matcher.final_score(exp_year=resume["experience_years"])
    except Exception as exc:
        logger.exception("JD matching failed")
        raise HTTPException(status_code=500, detail="Failed to score resume against JD") from exc


    recommendations = to_builtin(result)
    for recommendation in recommendations:
        if "score" in recommendation:
            recommendation["score"] = min(recommendation["score"], 100)

    return JDMatchResponse(
        recommendations=recommendations,
        missing_sections=missing_sections,
    )


@app.post("/ats", response_model=ATSResponse)
def ats_score_resume(payload: ResumeRequest) -> ATSResponse:
    safe_resume_url = validate_resume_url(payload.resume_url)

    try:
        result = extractor.ATS_score(
            pdf_url=safe_resume_url,
            download_timeout=DOWNLOAD_TIMEOUT_SECONDS,
            max_pdf_bytes=MAX_PDF_BYTES,
        )
    except ValueError as exc:
        logger.warning("ATS parsing validation failed for URL: %s", payload.resume_url)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("ATS scoring failed for URL: %s", payload.resume_url)
        raise HTTPException(status_code=400, detail="Failed to compute ATS score") from exc

    return ATSResponse(**to_builtin(result))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
