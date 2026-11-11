"""Paired bootstrap of the mR difference RIP-fixed - B1 at epoch 0.

Epoch-0 evaluation is deterministic, so seed variance does not apply; the
remaining uncertainty is the finite test set. We resample test *images*
(with all their captions) with replacement, recompute both models' six
recalls on the same resample, and report the 95% interval of the mR gap.

Usage: python scripts/bootstrap_epoch0.py --emb results/a3/epoch0_rsicd_emb.pt --n 2000 --out results/a3/bootstrap_rsicd.json
"""
import argparse
import json
import os

import numpy as np
import torch


def hits(img, txt, owner, ks=(1, 5, 10)):
    s = txt @ img.T
    n_img = img.shape[0]
    r_i = s.T.argsort(dim=1, descending=True)
    i2t = torch.stack([(owner[r_i[:, :k]] == torch.arange(n_img)[:, None]).any(1) for k in ks], 1)
    r_t = s.argsort(dim=1, descending=True)
    t2i = torch.stack([(r_t[:, :k] == owner[:, None]).any(1) for k in ks], 1)
    return i2t.float().numpy(), t2i.float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    e = torch.load(a.emb)
    owner = e["owner"].long()
    res = {}
    per = {}
    for name in ("B1", "RIP-fixed"):
        per[name] = hits(e[name], e["text"], owner)
    n_img = e["B1"].shape[0]
    caps_of = [np.where(owner.numpy() == i)[0] for i in range(n_img)]
    rng = np.random.default_rng(a.seed)

    def mr(h_i2t, h_t2i, img_idx):
        t_idx = np.concatenate([caps_of[i] for i in img_idx])
        return (h_i2t[img_idx].mean(0).sum() + h_t2i[t_idx].mean(0).sum()) / 6 * 100

    full = {k: mr(*per[k], np.arange(n_img)) for k in per}
    diffs = []
    for _ in range(a.n):
        idx = rng.integers(0, n_img, n_img)
        diffs.append(mr(*per["RIP-fixed"], idx) - mr(*per["B1"], idx))
    diffs = np.array(diffs)
    res = {"emb": os.path.basename(a.emb), "n_boot": a.n, "mR_B1": round(float(full["B1"]), 2), "mR_RIP": round(float(full["RIP-fixed"]), 2),
           "diff": round(float(full["RIP-fixed"] - full["B1"]), 2),
           "ci95": [round(float(np.percentile(diffs, 2.5)), 2), round(float(np.percentile(diffs, 97.5)), 2)],
           "boot_std": round(float(diffs.std(ddof=1)), 2),
           "p_diff_le_0": round(float((diffs <= 0).mean()), 4)}
    json.dump(res, open(a.out, "w"), indent=2)
    print(res)


if __name__ == "__main__":
    main()
