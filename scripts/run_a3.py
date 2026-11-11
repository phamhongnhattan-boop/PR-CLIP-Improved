"""Run one A3 experiment with the patched authors' code (ADMIS-TONGJI/PR-CLIP @ 242fba6).

Configurations (module flags are the ONLY difference between runs):
    B1 = no added module          R2 = RIP-fixed
    R4 = RIP-fixed + E1           R3 = RIP-fixed + E1 + EXT v2

Two modes:
  eval  : no training. Evaluate the RET-2 checkpoint (epoch 0) or a trained
          checkpoint (--resume) on one CSV. Deterministic, so one run is enough.
  train : one epoch on the RET-2 train split, evaluation on the held-out
          validation split before (epoch 0) and after (epoch 1) training.
          Checkpoints are written so the test set can be evaluated afterwards.

Every evaluation appends a line to <logs>/<name>/metrics.jsonl (see apply_patches.py).

Examples:
  python scripts/run_a3.py eval  --cfg R2 --data rsicd_test --repo /content/PR-CLIP --logs /content/drive/MyDrive/a3_logs
  python scripts/run_a3.py train --cfg R3 --seed 2 --repo /content/PR-CLIP --logs /content/drive/MyDrive/a3_logs
  python scripts/run_a3.py eval  --cfg R3 --seed 2 --data rsicd_test --resume <ckpt> --repo ... --logs ...
"""
import argparse
import os
import subprocess
import sys

CONFIGS = {
    "B1": {},
    "R2": {"PRCLIP_RIP": "1"},
    "R4": {"PRCLIP_RIP": "1", "PRCLIP_E1": "1"},
    "R3": {"PRCLIP_RIP": "1", "PRCLIP_E1": "1", "PRCLIP_EXT2": "1"},
}
DATA = {
    "rsicd_test": "./dataset/test/rsicd_test.csv",
    "rsitmd_test": "./dataset/test/rsitmd_test.csv",
    "val": "./dataset/val/ret2_val.csv",
}

# Shared setup, identical for every configuration (report Sec. IV-B)
COMMON = [
    "--images-dir", "./images", "--retrieval-images-dir", "./images",
    "--csv-separator", ",", "--csv-img-key", "filename",
    "--retrieval-csv-separator", ",", "--retrieval-csv-img-key", "filename",
    "--image-model", "ViT-B-32", "--image-model-builder", "openclip",
    "--text-model", "ViT-B-32", "--text-model-builder", "openclip",
    "--pretrained-image-model", "--pretrained-text-model",
    "--load-clip-weights", "--load-clip-weights-from", "./cache/weights/RET-2.pt",
    "--loss", "InfoNCE", "--batch-size", "16", "--workers", "2", "--precision", "amp",
]
TRAIN = [
    "--train-data", "./dataset/train/ret2_trainsplit.csv",
    "--epochs", "1", "--lr", "2e-7", "--warmup", "100", "--weight_decay", "0.5",
    "--max-grad-norm", "50", "--retrieval-frequency", "1", "--save-frequency", "1",
    "--save-most-recent", "--eval-first",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "train"])
    ap.add_argument("--cfg", required=True, choices=list(CONFIGS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data", default="val", choices=list(DATA))
    ap.add_argument("--resume", default=None, help="checkpoint to evaluate (eval mode)")
    ap.add_argument("--repo", default="/content/PR-CLIP")
    ap.add_argument("--logs", default="/content/drive/MyDrive/a3_logs")
    ap.add_argument("--images", default="./images", help="folder with all RSICD/RSITMD/RET-2 images")
    ap.add_argument("--ckpt", default="./cache/weights/RET-2.pt", help="RET-2 checkpoint")
    ap.add_argument("--cache-dir", default=None, help="open_clip cache (OpenAI ViT-B-32 weights)")
    ap.add_argument("--workers", default="2")
    ap.add_argument("--precision", default="amp", help="amp on GPU; fp32 only for CPU smoke tests")
    a = ap.parse_args()

    env = {**os.environ, "PRCLIP_RIP": "0", "PRCLIP_E1": "0", "PRCLIP_EXT2": "0", **CONFIGS[a.cfg]}
    if a.mode == "train":
        name = f"A3_{a.cfg}_train_s{a.seed}"
        extra = TRAIN + ["--retrieval-data", DATA["val"]]
    else:
        tag = "ep0" if a.resume is None else "ep1"
        name = f"A3_{a.cfg}_{tag}_{a.data}_s{a.seed}"
        extra = ["--retrieval-data", DATA[a.data], "--retrieval-frequency", "1"]
        if a.resume:
            extra += ["--resume", a.resume]
    common = [a.images if c == "./images" else a.ckpt if c == "./cache/weights/RET-2.pt"
              else a.workers if (i > 0 and COMMON[i - 1] == "--workers")
              else a.precision if (i > 0 and COMMON[i - 1] == "--precision") else c
              for i, c in enumerate(COMMON)]
    if a.cache_dir:
        common += ["--cache-dir", a.cache_dir]
    cmd = [sys.executable, "-m", "training.main", *common, *extra,
           "--seed", str(a.seed), "--name", name, "--logs", a.logs]
    print("RUN", a.cfg, {k: env[k] for k in ("PRCLIP_RIP", "PRCLIP_E1", "PRCLIP_EXT2")})
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=a.repo, env=env, check=True)


if __name__ == "__main__":
    main()
