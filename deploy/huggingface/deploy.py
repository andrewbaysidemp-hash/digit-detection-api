"""Deploy the API to a Hugging Face Docker Space (free CPU hosting).

    set HF_TOKEN=hf_...            # a token with write access (PowerShell: $env:HF_TOKEN="hf_...")
    set HF_SPACE=<user>/digit-api  # the Space id; it is created if it does not exist
    python -m pip install huggingface_hub
    python deploy/huggingface/deploy.py

Uploads only what the container needs (Dockerfile, server code, static page,
ONNX model, Space README) and prints the live URL. The same script runs in
.github/workflows/deploy-hf-space.yml on every push to main.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = ["Dockerfile", ".dockerignore", "requirements-server.txt", "server.py", "preprocess.py", "predictor.py", "detect.py"]


def main() -> int:
    token = os.environ.get("HF_TOKEN", "").strip()
    space = os.environ.get("HF_SPACE", "").strip()
    if not token or not space or "/" not in space:
        print("set HF_TOKEN (write token) and HF_SPACE (<user>/<space-name>) first", file=sys.stderr)
        return 2
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)

    stage = Path(tempfile.mkdtemp(prefix="hf-space-"))
    for name in FILES:
        shutil.copy(ROOT / name, stage / name)
    shutil.copytree(ROOT / "static", stage / "static")
    (stage / "models").mkdir()
    shutil.copy(ROOT / "models" / "mnist_cnn.onnx", stage / "models" / "mnist_cnn.onnx")
    shutil.copy(ROOT / "deploy" / "huggingface" / "README.md", stage / "README.md")

    sha = os.environ.get("GIT_SHA", "")[:7]
    api.upload_folder(folder_path=str(stage), repo_id=space, repo_type="space", commit_message=f"deploy {sha}".strip(), delete_patterns=["*.py", "static/*"])
    user, name = space.split("/", 1)
    print(f"uploaded. Space: https://huggingface.co/spaces/{space}")
    print(f"live URL (after the build finishes, 3-6 min): https://{user}-{name}.hf.space".replace("_", "-"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
