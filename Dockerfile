# Use the same Debian image where all pre-built wheels work,
# and add awslambdaric for Lambda compatibility.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CACHE_DIR=/app/app/.model_cache \
    HF_HOME=/app/app/.model_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/app/.model_cache \
    TRANSFORMERS_CACHE=/app/app/.model_cache \
    HF_HUB_DISABLE_XET=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install awslambdaric

COPY app ./app



# Pre-download the model so cold starts don't need network access.
# HF_HUB_DISABLE_XET avoids CI failures from Hugging Face's Xet/CAS download path.
RUN mkdir -p "$MODEL_CACHE_DIR" && \
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python app/cache_model.py

# Runtime should use the baked model cache only.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Lambda Runtime Interface Client as entrypoint
ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["app.main.handler"]
