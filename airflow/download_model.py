#!/usr/bin/env python3
"""
download_model.py — One-off host-side script to download the sentiment model.

Run this on your HOST (not inside Docker), where you have a modern Python:

    pip install huggingface_hub
    python download_model.py

This downloads the model files into airflow/data/hf_model/, which is mounted
into the airflow-worker container at /opt/airflow/data/hf_model. The ML
scoring script then loads from that local path, bypassing the broken
HuggingFace HTTP path that older transformers can't handle.

Re-running is safe — already-downloaded files are skipped.
"""

import os
from huggingface_hub import snapshot_download

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "data", "hf_model")

print(f"Downloading {MODEL}")
print(f"  -> {TARGET}")
print("(~500MB; may take 1-3 minutes on a typical connection.)")

snapshot_download(
    repo_id=MODEL,
    local_dir=TARGET,
    local_dir_use_symlinks=False,
)

print()
print("Done. Files in:")
for f in sorted(os.listdir(TARGET)):
    size_mb = os.path.getsize(os.path.join(TARGET, f)) / 1024 / 1024
    print(f"  {f:<40} {size_mb:>8.1f} MB")
