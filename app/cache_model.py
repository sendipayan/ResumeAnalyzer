import os
import platform
import sys
import traceback
from importlib import metadata


MODEL_NAME = "all-MiniLM-L6-v2"


def _version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not installed"


def main() -> None:
    cache_dir = os.environ.get("MODEL_CACHE_DIR", "/app/app/.model_cache")

    print("=== model cache build diagnostics ===", flush=True)
    print(f"python={sys.version}", flush=True)
    print(f"platform={platform.platform()}", flush=True)
    print(f"model={MODEL_NAME}", flush=True)
    print(f"MODEL_CACHE_DIR={cache_dir}", flush=True)
    print(f"HF_HOME={os.environ.get('HF_HOME')}", flush=True)
    print(f"SENTENCE_TRANSFORMERS_HOME={os.environ.get('SENTENCE_TRANSFORMERS_HOME')}", flush=True)
    print(f"TRANSFORMERS_CACHE={os.environ.get('TRANSFORMERS_CACHE')}", flush=True)
    print(f"HF_HUB_DISABLE_XET={os.environ.get('HF_HUB_DISABLE_XET')}", flush=True)
    print(f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}", flush=True)
    print(f"TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')}", flush=True)
    print(f"sentence-transformers={_version('sentence-transformers')}", flush=True)
    print(f"transformers={_version('transformers')}", flush=True)
    print(f"huggingface-hub={_version('huggingface-hub')}", flush=True)
    print(f"torch={_version('torch')}", flush=True)

    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(MODEL_NAME, cache_folder=cache_dir)
        print("model cached successfully", flush=True)
    except Exception:
        print("model cache failed with traceback:", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
