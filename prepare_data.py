"""
Merges every downloaded kas_Arab.tsv file (from multiple BPCC subfolders)
into one deduplicated train/val CSV pair.

Column order isn't assumed - each row is checked for Perso-Arabic script
characters and assigned to the 'kashmiri' column regardless of which
position it was in, so a wrong assumption about file layout doesn't
silently corrupt the data.

Usage:
    python prepare_data.py --in-dir ./bpcc_kashmiri --out-dir ./data --val-frac 0.02
"""
import argparse
import csv
import random
import unicodedata
from pathlib import Path

# Perso-Arabic block ranges (covers Kashmiri Nastaliq)
ARABIC_RANGES = [(0x0600, 0x06FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF), (0x0750, 0x077F)]


def is_perso_arabic(text: str) -> bool:
    if not text:
        return False
    hits = sum(
        1
        for ch in text
        if any(lo <= ord(ch) <= hi for lo, hi in ARABIC_RANGES)
    )
    return hits / max(len(text), 1) > 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="./bpcc_kashmiri")
    ap.add_argument("--out-dir", default="./data")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = list(Path(args.in_dir).rglob("*kas_Arab*"))
    files = [f for f in files if f.suffix in (".tsv", ".txt", ".csv")]
    print(f"Found {len(files)} source file(s): {[str(f) for f in files]}")

    pairs = []
    for f in files:
        with open(f, encoding="utf-8", errors="ignore") as fh:
            reader = csv.reader(fh, delimiter="\t")
            for row in reader:
                if len(row) < 2:
                    continue
                a, b = row[0].strip(), row[1].strip()
                if not a or not b:
                    continue
                # Assign by script detection, not by column position
                if is_perso_arabic(a) and not is_perso_arabic(b):
                    eng, kas = b, a
                elif is_perso_arabic(b) and not is_perso_arabic(a):
                    eng, kas = a, b
                else:
                    continue  # ambiguous row (e.g. header), skip
                pairs.append((eng, kas))

    before = len(pairs)
    pairs = list(dict.fromkeys(pairs))  # dedupe, preserve order
    print(f"{before:,} rows read, {len(pairs):,} unique pairs after dedup")

    random.seed(args.seed)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_frac))
    val, train = pairs[:n_val], pairs[n_val:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", train), ("val", val)]:
        path = out_dir / f"{name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["english", "kashmiri"])
            writer.writerows(rows)
        print(f"Wrote {path} ({len(rows):,} rows)")


if __name__ == "__main__":
    main()
