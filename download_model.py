"""Download distilbert-base-multilingual-cased with retry logic."""
import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"

from huggingface_hub import snapshot_download

MODEL_NAME = "distilbert-base-multilingual-cased"
print(f"Downloading {MODEL_NAME}...")
print("This may take several minutes on slow connections...")

try:
    path = snapshot_download(
        MODEL_NAME,
        resume_download=True,
    )
    print(f"OK - Downloaded to: {path}")
except Exception as e:
    print(f"FAIL: {e}")
    print("Retrying individual files...")
    from huggingface_hub import hf_hub_download
    files = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]
    for f in files:
        try:
            print(f"  Downloading {f}...")
            hf_hub_download(MODEL_NAME, f, resume_download=True)
            print(f"  OK: {f}")
        except Exception as e2:
            print(f"  FAIL {f}: {e2}")
