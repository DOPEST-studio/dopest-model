"""Publish the release (adapter, calibration, model card) to a Hugging Face model repository.

    op run --env-file=scripts/hf.op.env -- uv run python scripts/publish_hf.py --repo dopest-studio/dopest-1.0-31b

The token comes from the environment (HF_TOKEN via `op run`), never from a file or an argument. Files are
uploaded from release/: README.md, calibration.json, adapter/adapter_config.json, adapter/adapter_model.safetensors.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
FILES = {"README.md": "README.md", "calibration.json": "calibration.json",
         "adapter/adapter_config.json": "adapter_config.json",
         "adapter/adapter_model.safetensors": "adapter_model.safetensors"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="<owner>/<name> on huggingface.co")
    parser.add_argument("--private", action="store_true", help="create the repo private (flip to public later)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN missing: run under `op run --env-file=scripts/hf.op.env -- ...`")
    api = HfApi(token=token)
    for local in FILES:
        if not (ROOT / "release" / local).exists():
            raise SystemExit(f"missing release/{local}")
    if args.dry_run:
        print("would create", args.repo, "and upload", list(FILES.values()))
        return
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    for local, remote in FILES.items():
        api.upload_file(path_or_fileobj=str(ROOT / "release" / local), path_in_repo=remote, repo_id=args.repo,
                        commit_message=f"add {remote}")
        print("uploaded", remote)
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
