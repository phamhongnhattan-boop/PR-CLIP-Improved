"""Standalone epoch-0 evaluation: RET-2 checkpoint with / without RIP-fixed.

Needs only torch + open_clip_torch==2.20.0 (no ITRA code). At epoch 0 the
PR-CLIP retrieval path is exactly the CLIP backbone (projection heads are
identity), so this reproduces evaluation/retrieval.py of the authors' code:
  * open_clip ViT-B-32 'openai' transforms, weights replaced by RET-2.pt
  * images de-duplicated, all captions of an image are positives
  * I2T R@K = 1 if any caption of the image is in the top K; T2I one positive
  * mR = mean of the six recalls

Usage:
  python scripts/eval_epoch0_standalone.py --ckpt RET-2.pt --csv rsicd_test.csv \
      --images images/ --mode both --out results/a3/epoch0_rsicd_test.json
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prclip_patches"))
from a3_modules import rip_pool  # noqa: E402


def load_model(ckpt, device):
    # pretrained="openai" in the authors' code builds the OpenAI model, which uses QuickGELU
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None, device=device,
                                                                 force_quick_gelu=True)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    msg = model.load_state_dict(sd, strict=True)
    model.eval()
    return model, preprocess, str(msg)


def load_csv(path):
    enc = "gb18030" if "rsicd" in os.path.basename(path) else None
    df = pd.read_csv(path, encoding=enc)
    images, idx = np.unique(df["filename"].tolist(), return_index=True)
    caps, owner = [], []
    for i, img in enumerate(images):
        for c in df.loc[df["filename"] == img, "title"].tolist():
            caps.append(c)
            owner.append(i)
    return list(images), caps, np.array(owner)


@torch.no_grad()
def encode_images(model, preprocess, paths, rip, bs, device):
    out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([preprocess(Image.open(p)) for p in paths[i:i + bs]]).to(device)
        f = rip_pool(model.encode_image, x) if rip else model.encode_image(x)
        out.append(F.normalize(f.float(), dim=-1).cpu())
        print(f"  images {min(i + bs, len(paths))}/{len(paths)}", flush=True)
    return torch.cat(out)


@torch.no_grad()
def encode_texts(model, caps, bs, device):
    out = []
    for i in range(0, len(caps), bs):
        t = open_clip.tokenize(caps[i:i + bs]).to(device)
        out.append(F.normalize(model.encode_text(t).float(), dim=-1).cpu())
    return torch.cat(out)


def recalls(img, txt, owner, ks=(1, 5, 10)):
    s = txt @ img.T                                  # [n_text, n_image]
    m = {}
    rank_i = s.T.argsort(dim=1, descending=True)      # per image: captions ranked
    for k in ks:
        top = rank_i[:, :k]
        m[f"image2text-R@{k}"] = float((torch.from_numpy(owner)[top] == torch.arange(len(img))[:, None]).any(1).float().mean() * 100)
    rank_t = s.argsort(dim=1, descending=True)
    for k in ks:
        m[f"text2image-R@{k}"] = float((rank_t[:, :k] == torch.from_numpy(owner)[:, None]).any(1).float().mean() * 100)
    m["mean-recall"] = float(np.mean(list(m.values())))
    return {k: round(v, 2) for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--mode", choices=["B1", "RIP", "both"], default="both")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-emb", default=None, help="save embeddings (.pt) for bootstrap tests")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, msg = load_model(a.ckpt, device)
    images, caps, owner = load_csv(a.csv)
    paths = [os.path.join(a.images, p) for p in images]
    print(f"{a.csv}: {len(images)} images, {len(caps)} captions, device={device}", flush=True)

    t0 = time.time()
    txt = encode_texts(model, caps, a.bs * 4, device)
    t_text = time.time() - t0
    embs = {"text": txt, "owner": torch.from_numpy(owner)}
    res = {"csv": os.path.basename(a.csv), "images": len(images), "captions": len(caps),
           "checkpoint": os.path.basename(a.ckpt), "load_state_dict": msg, "device": device,
           "cpu": platform.processor() or platform.machine(), "torch_threads": torch.get_num_threads(),
           "text_encode_s": round(t_text, 2), "runs": {}}
    for name, rip in [("B1", False), ("RIP-fixed", True)]:
        if a.mode not in ("both", "RIP" if rip else "B1"):
            continue
        t0 = time.time()
        img = encode_images(model, preprocess, paths, rip, a.bs, device)
        t_img = time.time() - t0
        t0 = time.time()
        m = recalls(img, txt, owner)
        embs[name] = img
        t_rank = time.time() - t0
        res["runs"][name] = {**m, "image_forward_passes": len(images) * (8 if rip else 1),
                             "image_encode_s": round(t_img, 2), "rank_s": round(t_rank, 2),
                             "inference_time_s": round(t_img + t_text + t_rank, 2)}
        print(name, res["runs"][name], flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=2)
    if a.save_emb:
        torch.save(embs, a.save_emb)


if __name__ == "__main__":
    main()
