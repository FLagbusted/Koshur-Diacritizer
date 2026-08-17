"""
Downloads only the Kashmiri (kas_Arab) files from ai4bharat/BPCC, across
every subfolder that has per-language splits. Skips the other 21
languages entirely, so this pulls MB, not the full 107 GB.

Requires: huggingface-cli login (or HF_TOKEN set) with the BPCC gate
already accepted at huggingface.co/datasets/ai4bharat/BPCC.

Usage:
    python download_kashmiri_bpcc.py --out ./bpcc_kashmiri
"""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "ai4bharat/BPCC"

# Every pattern that could contain a Kashmiri-tagged file, based on the
# repo's folder layout (daily, wiki, ilci, massive, nllb_seed,
# nllb_filtered, samanantar_v2, comparable, bpcc-seed-*).
ALLOW_PATTERNS = [
    "*/kas_Arab.tsv",
    "*/kas_Arab*.tsv",
    "*/*kas_Arab*",
    "*kas_Arab*",
    "README.md",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./bpcc_kashmiri")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    print("Downloading Kashmiri-only files from ai4bharat/BPCC ...")
    local_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=ALLOW_PATTERNS,
        local_dir=args.out,
    )
    print(f"Done. Files saved under: {local_path}")

    found = list(Path(args.out).rglob("*kas_Arab*"))
    if not found:
        print(
            "WARNING: no kas_Arab files matched. The gate may not be accepted "
            "yet, or the folder layout differs from what this script assumes. "
            "Run: huggingface-cli download ai4bharat/BPCC --repo-type dataset "
            "--local-dir /tmp/bpcc_list (small metadata only) and inspect "
            "manually if this happens."
        )
    else:
        print(f"Found {len(found)} Kashmiri file(s):")
        for f in found:
            print(f"  {f}  ({f.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
