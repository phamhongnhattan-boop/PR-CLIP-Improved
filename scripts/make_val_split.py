"""Create an image-level validation split from RET-2.

Why: in the authors' repo, dataset/test/rsicd_val.csv is byte-identical to
rsicd_test.csv (same SHA-256), and 733 of the 856 rsitmd_val images are also in
ret2.csv. Neither can be used to choose epochs or checkpoints.

This script holds out --val-frac of RET-2 *images* (all 5 captions together),
stratified by scene prefix (e.g. "airport_12.jpg" -> "airport"), and writes:
    dataset/train/ret2_trainsplit.csv
    dataset/val/ret2_val.csv
    dataset/split_manifest.json   (counts + SHA-256 of every CSV used)

Usage:  python scripts/make_val_split.py --repo /content/PR-CLIP --val-frac 0.1 --seed 0
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def scene(name: str) -> str:
    stem = name.rsplit(".", 1)[0]
    return stem.rsplit("_", 1)[0] if "_" in stem else "numeric"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    repo = Path(a.repo)
    src = repo / "dataset/train/ret2.csv"
    df = pd.read_csv(src)

    groups = defaultdict(list)
    for img in sorted(df["filename"].unique()):
        groups[scene(img)].append(img)
    rng = random.Random(a.seed)
    val_imgs = set()
    for _, imgs in sorted(groups.items()):
        rng.shuffle(imgs)
        val_imgs.update(imgs[: max(1, round(len(imgs) * a.val_frac))])

    val = df[df["filename"].isin(val_imgs)]
    train = df[~df["filename"].isin(val_imgs)]
    out_tr = repo / "dataset/train/ret2_trainsplit.csv"
    out_va = repo / "dataset/val/ret2_val.csv"
    out_va.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_tr, index=False)
    val.to_csv(out_va, index=False)

    tests = {p.name: p for p in (repo / "dataset/test").glob("*.csv")}
    leak = {n: len(set(pd.read_csv(p)["filename"]) & val_imgs) for n, p in tests.items()}
    assert not set(train["filename"]) & val_imgs
    manifest = {
        "source": str(src), "seed": a.seed, "val_frac": a.val_frac,
        "train_images": int(train["filename"].nunique()), "train_pairs": len(train),
        "val_images": int(val["filename"].nunique()), "val_pairs": len(val),
        "val_overlap_with_test_csvs": leak,
        "sha256": {p.name: sha256(p) for p in [src, out_tr, out_va, *tests.values()]},
        "note": "rsicd_val.csv == rsicd_test.csv in the authors' repo; never use it for model selection.",
    }
    (repo / "dataset/split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
