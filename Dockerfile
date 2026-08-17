# Use the same Debian image where all pre-built wheels work,
# and add awslambdaric for Lambda compatibility.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0 \
    MODEL_CACHE_DIR=/app/app/.model_cache

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install awslambdaric

COPY app ./app



# Pre-download the model so cold starts don't need network access
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/app/app/.model_cache')"

# Lambda Runtime Interface Client as entrypoint
ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["app.main.handler"]
