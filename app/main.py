import ipaddress
import asyncio
import logging
import os
import socket
import ast
import time
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import pickle
import requests
from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import AnyHttpUrl, BaseModel, Field
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Keep logs minimal: app lifecycle + route logs (via uvicorn access logger).
for _name in (
    
    "torch",
    "uvicorn",
    "uvicorn.error",
    "fastapi",
    "starlette",
    "httpx",
    "urllib3",
):
    logging.getLogger(_name).setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
PICKLE_PATH= BASE_DIR/ "dataset" / "job_embeddings.pkl"
JOB_DATA_PATH = BASE_DIR / "dataset" / "job_data.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
def _resolve_model_cache_dir() -> Path:
    override = os.getenv("MODEL_CACHE_DIR", "").strip()
    if override:
        return Path(override)

    candidates = [
        BASE_DIR / ".model_cache",
        Path("/tmp") / "model_cache",
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            with open(probe, "w", encoding="utf-8") as handle:
                handle.write("ok")
            probe.unlink(missing_ok=True)
            return candidate
        except Exception:
            continue

    # Fall back to the base directory (may be read-only in Lambda).
    return BASE_DIR / ".model_cache"


MODEL_CACHE_DIR = _resolve_model_cache_dir()
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(MODEL_CACHE_DIR)
if "SENTENCE_TRANSFORMERS_HOME" not in os.environ:
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(MODEL_CACHE_DIR)
if "TRANSFORMERS_CACHE" not in os.environ:
    os.environ["TRANSFORMERS_CACHE"] = str(MODEL_CACHE_DIR)
MODEL_REVISION = os.getenv("MODEL_REVISION", "").strip() or None

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
ALLOW_HTTP_RESUME_URL = os.getenv(
    "ALLOW_HTTP_RESUME_URL", "true" if APP_ENV != "production" else "false"
).strip().lower() == "true"
ALLOW_PRIVATE_RESUME_HOSTS = os.getenv("ALLOW_PRIVATE_RESUME_HOSTS", "false").strip().lower() == "true"
DOWNLOAD_TIMEOUT_SECONDS = int(os.getenv("DOWNLOAD_TIMEOUT_SECONDS", "20"))
MAX_PDF_MB = int(os.getenv("MAX_PDF_MB", "5"))
MAX_PDF_BYTES = MAX_PDF_MB * 1024 * 1024
MAX_RSS_MB = int(os.getenv("MAX_RSS_MB", "800"))
RETRY_AFTER_SECONDS = int(os.getenv("RETRY_AFTER_SECONDS", "30"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "4"))
MAX_QUEUE_WAIT_SECONDS = int(os.getenv("MAX_QUEUE_WAIT_SECONDS", "30"))
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").strip().lower() == "true"
API_KEY = os.getenv("API_KEY")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "").strip()
EMBEDDING_TIMEOUT_SECONDS = int(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30"))
PER_IP_RATE_LIMIT_MAX = int(os.getenv("PER_IP_RATE_LIMIT_MAX", "20"))
PER_IP_RATE_LIMIT_WINDOW_SECONDS = float(os.getenv("PER_IP_RATE_LIMIT_WINDOW_SECONDS", "60"))
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
logger.debug("ALLOWED_ORIGINS=%s", ALLOWED_ORIGINS)

if MAX_PDF_MB <= 0:
    MAX_PDF_MB = 5
    MAX_PDF_BYTES = MAX_PDF_MB * 1024 * 1024

# Request-count rate limiting is disabled.
_request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS) if MAX_CONCURRENT_REQUESTS > 0 else None
_rate_limit_lock = asyncio.Lock()
_per_ip_request_times: dict[str, deque] = {}


@lru_cache(maxsize=1)
def _np():
    import numpy as np
    return np


@lru_cache(maxsize=1)
def _pd():
    import pandas as pd
    return pd


@lru_cache(maxsize=1)
def _proc():
    try:
        import psutil
    except Exception:
        return None
    return psutil.Process()


def _get_cgroup_memory_mb() -> float | None:
    # Prefer cgroup memory to align with Docker/EC2 enforcement.
    candidates = (
        "/sys/fs/cgroup/memory.current",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",  # cgroup v1
    )
    for path in candidates:
        try:
            text = Path(path).read_text(encoding="utf-8").strip()
            if text:
                return int(text) / (1024 * 1024)
        except Exception:
            continue
    return None


def _is_over_memory_limit() -> tuple[bool, float | None]:
    if MAX_RSS_MB <= 0:
        return False, None
    cgroup_mb = _get_cgroup_memory_mb()
    if cgroup_mb is not None:
        return cgroup_mb >= MAX_RSS_MB, cgroup_mb
    proc = _proc()
    if proc is None:
        return False, None
    try:
        rss_mb = proc.memory_info().rss / (1024 * 1024)
    except Exception:
        return False, None
    return rss_mb >= MAX_RSS_MB, rss_mb


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
        if ip:
            return ip
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


async def _enforce_rate_limits(request: Request) -> JSONResponse | None:
    if PER_IP_RATE_LIMIT_MAX <= 0:
        return None

    now = time.monotonic()
    ip = _get_client_ip(request)

    async with _rate_limit_lock:
        # Per-IP limiter: 20 req / 60 sec default.
        if PER_IP_RATE_LIMIT_MAX > 0 and PER_IP_RATE_LIMIT_WINDOW_SECONDS > 0:
            window = _per_ip_request_times.setdefault(ip, deque())
            cutoff = now - PER_IP_RATE_LIMIT_WINDOW_SECONDS
            while window and window[0] < cutoff:
                window.popleft()
            if len(window) >= PER_IP_RATE_LIMIT_MAX:
                retry_after = max(1, math.ceil(window[0] + PER_IP_RATE_LIMIT_WINDOW_SECONDS - now))
                headers = {"Retry-After": str(retry_after)}
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded (per IP). Please retry later."},
                    headers=headers,
                )
            window.append(now)

    return None


class RemoteEmbedder:
    def __init__(self, endpoint_url: str, timeout_seconds: int = 30):
        if not endpoint_url:
            raise ValueError("EMBEDDING_API_URL is not configured")
        self.endpoint_url = endpoint_url
        self.timeout_seconds = timeout_seconds

    def encode(self, texts):
        if isinstance(texts, str):
            payload = [texts]
            single = True
        else:
            payload = ["" if t is None else str(t) for t in (texts or [])]
            single = False

        if not payload:
            return [] if not single else []

        response = requests.post(
            self.endpoint_url,
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return data[0] if single else data


@lru_cache(maxsize=1)
def get_extractor():
    try:
        from app.services.resume_parser import CVPipeline
    except ModuleNotFoundError:
        from services.resume_parser import CVPipeline
    return CVPipeline()



@lru_cache(maxsize=1)
def get_model() -> Any:
    from sentence_transformers import SentenceTransformer
    kwargs = {"cache_folder": str(MODEL_CACHE_DIR)}
    if MODEL_REVISION:
        kwargs["revision"] = MODEL_REVISION
    try:
        return SentenceTransformer(MODEL_NAME, **kwargs)
    except TypeError:
        if "revision" in kwargs:
            logger.warning("MODEL_REVISION is set but not supported; ignoring revision.")
            kwargs.pop("revision", None)
            return SentenceTransformer(MODEL_NAME, **kwargs)
        raise


@lru_cache(maxsize=1)
def get_job_df() -> Any:
    pd = _pd()
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

def get_pickle()->Any:
    if not PICKLE_PATH.exists():
        raise FileNotFoundError(f"Pickle not found at: {PICKLE_PATH}")
    with open(PICKLE_PATH, "rb") as f:
      store = pickle.load(f)

    return store


def to_builtin(value):
    if isinstance(value, dict):  
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    np = _np()
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


@asynccontextmanager
async def _lifespan(_: FastAPI):
    logger.info("App starting up")
    try:
        yield
    finally:
        logger.info("App shutting down")


app = FastAPI(
    title="Resume Role Predictor API",
    description="Parse resume from URL and return ranked role recommendations.",
    version="1.2.0",
    lifespan=_lifespan,
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
    if request.url.path not in {"/health", "/", "/docs"}:
        rate_limit_response = await _enforce_rate_limits(request)
        if rate_limit_response is not None:
            return rate_limit_response

    if request.url.path in {"/score", "/jdmatch", "/ats", "/feeback"}:
        if REQUIRE_API_KEY:
            if not API_KEY:
                logger.error("REQUIRE_API_KEY is enabled but API_KEY is not set")
                return JSONResponse(status_code=503, content={"detail": "Service authentication is not configured"})
            provided_key = request.headers.get("x-api-key")
            if not provided_key or provided_key != API_KEY:
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

        acquired = False
        if _request_semaphore is not None:
            wait_seconds = max(MAX_QUEUE_WAIT_SECONDS, 0)
            try:
                await asyncio.wait_for(_request_semaphore.acquire(), timeout=wait_seconds)
                acquired = True
            except asyncio.TimeoutError:
                headers = {"Retry-After": str(max(RETRY_AFTER_SECONDS, 0))}
                return JSONResponse(
                    status_code=503,
                    content={"detail": "Service is busy. Please retry later."},
                    headers=headers,
                )

        try:
            return await call_next(request)
        finally:
            if acquired and _request_semaphore is not None:
                _request_semaphore.release()

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
    ATS_score: float
    section_score: float
    contact_score: float
    formating_score: float
    issues: list[str] = Field(default_factory=list)


class FeedbackResponse(BaseModel):
    recommendations: list[dict[str, Any]]
    ats: ATSResponse


@app.get("/health")
def health_check() -> dict:
    model_loaded = True
    dataset_loaded = True
    pickle_loaded= True
    embedding_stats: dict[str, Any] | None = None
    memory_stats: dict[str, Any] = {
        "max_rss_mb": MAX_RSS_MB,
        "retry_after_seconds": RETRY_AFTER_SECONDS,
    }
    concurrency_stats: dict[str, Any] = {
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "max_queue_wait_seconds": MAX_QUEUE_WAIT_SECONDS,
    }
    model_stats: dict[str, Any] = {
        "name": MODEL_NAME,
        "revision": MODEL_REVISION,
    }
    errors: list[str] = []

    try:
        model = get_model()
        dummy_text = "health-check"
        start = time.monotonic()
        embedding = model.encode(dummy_text)
        elapsed = time.monotonic() - start
        if isinstance(embedding, (list, tuple)) and embedding:
            emb_len = len(embedding[0])
            emb_sample = list(embedding[0][:3])
        else:
            emb_len = None
            emb_sample = None
        embedding_stats = {
            "dummy_text": dummy_text,
            "duration_seconds": round(elapsed, 4),
            "embedding_length": emb_len,
            "embedding_sample": emb_sample,
        }
    except Exception as exc:
        model_loaded = False
        errors.append(f"model_error: {exc}")



    try:
        get_job_df()
    except Exception as exc:
        dataset_loaded = False
        errors.append(f"dataset_error: {exc}")

    try:
        get_pickle()
    except Exception as exc:
        pickle_loaded = False
        errors.append(f"pickle_error: {exc}")

    proc = _proc()
    if proc is not None:
        try:
            memory_stats["current_rss_mb"] = round(proc.memory_info().rss / (1024 * 1024), 2)
        except Exception:
            memory_stats["current_rss_mb"] = None
    cgroup_mb = _get_cgroup_memory_mb()
    memory_stats["current_cgroup_mb"] = round(cgroup_mb, 2) if cgroup_mb is not None else None

    status = "healthy" if model_loaded and dataset_loaded and pickle_loaded else "degraded"
    return {
        "status": status,
        "model": MODEL_NAME,
        "model_loaded": model_loaded,
        "job_data_loaded": dataset_loaded,
        "Embeddings_loaded": pickle_loaded,
        "embedding_stats": embedding_stats,
        "api_key_required": REQUIRE_API_KEY,
        "rate_limit": {
            "enabled": True,
            "best_effort": True,
            "scope": "in-memory per instance",
            "per_ip_max_requests": PER_IP_RATE_LIMIT_MAX,
            "per_ip_window_seconds": PER_IP_RATE_LIMIT_WINDOW_SECONDS,
        },
        "memory_limit": memory_stats,
        "concurrency_limit": concurrency_stats,
        "model_info": model_stats,
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
            "/feeback": "Recommendations + ATS score (POST)",
            "/docs": "Swagger UI",
        },
    }


@app.post("/score", response_model=ScoreResponse)
def score_resume(payload: ResumeRequest) -> ScoreResponse:
    try:
        from app.services.score import RolePredicter
        from app.services.jobs import JobRedirectBuilder
    except ModuleNotFoundError:
        from services.score import RolePredicter
        from services.jobs import JobRedirectBuilder

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
        store = get_pickle()
    except Exception as exc:
        logger.exception("Pickle load failed")
        raise HTTPException(status_code=503, detail="Embeddings not available") from exc
    
    try:
        resume = get_extractor().run(
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
            store=store,
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
    try:
        from app.services.jdmatch import JDMatch
    except ModuleNotFoundError:
        from services.jdmatch import JDMatch

    safe_resume_url = validate_resume_url(payload.resume_url)

    try:
        model = get_model()
    except Exception as exc:
        logger.exception("Model load failed")
        raise HTTPException(status_code=503, detail="Model not available") from exc
    
    try:
        store = get_pickle()
    except Exception as exc:
        logger.exception("Pickle load failed")
        raise HTTPException(status_code=503, detail="Embeddings not available") from exc

    try:
        resume = get_extractor().run(
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
            store=store,
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
        result = get_extractor().ATS_score(
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


@app.post("/feedback", response_model=FeedbackResponse)
def feedback_resume(payload: ResumeRequest) -> FeedbackResponse:
    try:
        from app.services.score import RolePredicter
    except ModuleNotFoundError:
        from services.score import RolePredicter

    safe_resume_url = validate_resume_url(payload.resume_url)

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
        store = get_pickle()
    except Exception as exc:
        logger.exception("Pickle load failed")
        raise HTTPException(status_code=503, detail="Embeddings not available") from exc

    try:
        resume = get_extractor().run(
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

    try:
        scorer = RolePredicter(
            resume_skills=resume["skills_list"],
            job_df=job_df,
            store=store,
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

    recommendations = to_builtin(result)
    for recommendation in recommendations:
        if "score" in recommendation:
            recommendation["score"] = min(recommendation["score"], 100)

    try:
        ats_raw = get_extractor().ATS_score(
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

    return FeedbackResponse(
        recommendations=recommendations,
        ats=ATSResponse(**to_builtin(ats_raw)),
    )


from mangum import Mangum
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
