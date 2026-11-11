"""Aggregate A3 metrics.jsonl files into mean ± std tables (CSV, Markdown, LaTeX).

Reads every metrics.jsonl under --logs (run names produced by run_a3.py), then
  * epoch-0 test results (no training, deterministic),
  * epoch-1 test results per seed -> mean ± std,
  * validation-selected results: for each (cfg, seed) the epoch (0 or 1) with the
    higher validation mR is chosen, and only that model's test result is reported.

Outputs:
  results/a3/all_runs.csv, results/a3/summary.csv, results/a3/summary.md
  report/tables/a3_multiseed.tex  (\\input by the report)

Usage: python scripts/aggregate_results.py --logs /content/drive/MyDrive/a3_logs --out .
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

KEYS = ["retrieval-image2text-R@1", "retrieval-image2text-R@5", "retrieval-image2text-R@10",
        "retrieval-text2image-R@1", "retrieval-text2image-R@5", "retrieval-text2image-R@10",
        "retrieval-mean-recall"]
SHORT = ["I2T R@1", "I2T R@5", "I2T R@10", "T2I R@1", "T2I R@5", "T2I R@10", "mR"]
ORDER = ["B1", "R2", "R4", "R3"]
LABEL = {"B1": "B1 (baseline)", "R2": "R2 RIP-fixed", "R4": "R4 RIP+E1", "R3": "R3 RIP+E1+EXT v2"}
PAT = re.compile(r"A3_(?P<cfg>B1|R2|R3|R4)_(?P<kind>train|ep0|ep1)(?:_(?P<data>rsicd_test|rsitmd_test|val))?_s(?P<seed>\d+)")


def load(logs: Path) -> pd.DataFrame:
    rows = []
    for f in logs.rglob("metrics.jsonl"):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            m = PAT.search(r["name"])
            if not m:
                continue
            d = m.groupdict()
            data = d["data"] or Path(r["retrieval_data"]).stem.replace("ret2_val", "val")
            rows.append({"cfg": d["cfg"], "kind": d["kind"], "data": data, "seed": int(d["seed"]),
                         "epoch": int(r["epoch"]), "it_s": r.get("inference_time_s"),
                         **{s: r[k] for k, s in zip(KEYS, SHORT)}})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No A3 metrics found under {logs}")
    # keep the last evaluation if a run was repeated
    return df.drop_duplicates(["cfg", "kind", "data", "seed", "epoch"], keep="last")


def fmt(series: pd.Series) -> str:
    if len(series) == 1:
        return f"{series.iloc[0]:.2f}"
    return f"{series.mean():.2f} $\\pm$ {series.std(ddof=1):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--test", default="rsicd_test")
    a = ap.parse_args()
    out = Path(a.out)
    df = load(Path(a.logs))
    (out / "results/a3").mkdir(parents=True, exist_ok=True)
    df.sort_values(["cfg", "kind", "data", "seed"]).to_csv(out / "results/a3/all_runs.csv", index=False)

    test = df[df["data"] == a.test]
    ep0 = test[test["kind"] == "ep0"]
    ep1 = test[test["kind"] == "ep1"]
    val = df[(df["kind"] == "train") & (df["data"] == "val")]

    summary, tex = [], []
    for cfg in ORDER:
        blocks = {
            "epoch 0 (no training)": ep0[ep0["cfg"] == cfg],
            "epoch 1": ep1[ep1["cfg"] == cfg],
        }
        sel_rows = []
        for seed, g in val[val["cfg"] == cfg].groupby("seed"):
            if not {0, 1} <= set(g["epoch"]):
                continue
            best_ep = int(g.sort_values("mR").iloc[-1]["epoch"])
            src = ep0[ep0["cfg"] == cfg] if best_ep == 0 else ep1[(ep1["cfg"] == cfg) & (ep1["seed"] == seed)]
            if not src.empty:
                sel_rows.append(src.iloc[[0]].assign(seed=seed, selected_epoch=best_ep))
        blocks["val-selected"] = pd.concat(sel_rows) if sel_rows else pd.DataFrame(columns=SHORT)
        for setting, g in blocks.items():
            if g.empty:
                continue
            rec = {"cfg": cfg, "setting": setting, "n": len(g)}
            for s in SHORT:
                rec[s] = g[s].mean()
                rec[s + "_std"] = g[s].std(ddof=1) if len(g) > 1 else 0.0
            if "selected_epoch" in g:
                rec["selected_epochs"] = ",".join(str(int(e)) for e in g["selected_epoch"])
            rec["IT_s"] = g["it_s"].mean()
            summary.append(rec)
            tex.append(f"{LABEL[cfg]} & {setting} & {len(g)} & "
                       + " & ".join(fmt(g[s]) for s in SHORT) + " \\\\")

    sm = pd.DataFrame(summary)
    sm.to_csv(out / "results/a3/summary.csv", index=False)
    r = sm.round(2).astype(str)
    md = "| " + " | ".join(r.columns) + " |\n|" + "---|" * len(r.columns) + "\n"
    md += "\n".join("| " + " | ".join(str(x) for x in row) + " |" for row in r.values.tolist())
    (out / "results/a3/summary.md").write_text(md + "\n")
    tdir = out / "report/tables"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "a3_multiseed_rows.tex").write_text("\n".join(tex) + "\n")
    print(sm.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
