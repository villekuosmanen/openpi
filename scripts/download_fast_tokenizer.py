from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the FAST tokenizer into a local weights directory.")
    parser.add_argument(
        "--repo-id",
        default="physical-intelligence/fast",
        help="Hugging Face repo to download.",
    )
    parser.add_argument(
        "--output-dir",
        default="weights/fast",
        help="Directory where the tokenizer snapshot should be stored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(output_dir),
    )
    print(f"FAST tokenizer downloaded to {output_dir}")


if __name__ == "__main__":
    main()
