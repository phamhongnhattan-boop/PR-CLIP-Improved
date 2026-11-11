"""Extract training configuration and retrieval metrics from the saved outputs of
the notebooks in notebooks/ into results/legacy/<notebook>.json and
results/legacy/legacy_runs.csv.  Only values printed in notebook outputs are used.

Usage: python scripts/extract_notebook_logs.py
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
OUT = ROOT / "results/legacy"
KEYS = ["image2text-R@1", "image2text-R@5", "image2text-R@10",
        "text2image-R@1", "text2image-R@5", "text2image-R@10", "mean-recall"]
ARGS = ["name", "lr", "epochs", "batch_size", "seed", "load_clip_weights", "train_data", "retrieval_data", "precision"]


def out_text(o):
    if "text" in o:
        return "".join(o["text"])
    if "data" in o:
        return "".join(o["data"].get("text/plain", []))
    if "traceback" in o:
        return "\n".join(o["traceback"])
    return ""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = []
    for nb_path in sorted(NB.glob("*.ipynb")):
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        text = "\n".join(out_text(o) for c in nb["cells"] for o in c.get("outputs", []))
        text = text.replace("\\t", "\t").replace("\\n", "\n")
        runs = []
        for m in re.finditer(r"Namespace\((.*?)\)\s*$", text, re.M):
            ns = m.group(1)
            cfg = {}
            for k in ARGS:
                mm = re.search(rf"\b{k}=('([^']*)'|[^,]+)", ns)
                if mm:
                    cfg[k] = mm.group(2) if mm.group(2) is not None else mm.group(1)
            runs.append({"config": cfg, "evaluations": []})
        for m in re.finditer(r"Finished evaluation of \[(.*?)\] at epoch (\d+)", text):
            window = text[m.end(): m.end() + 1500].split("Finished evaluation")[0]
            vals = {}
            for k, v in re.findall(r"retrieval-([\w@-]+)\s+([\d.]+)", window):
                vals.setdefault(k, v)
            if "mean-recall" not in vals:
                continue
            ev = {"run": m.group(1), "epoch": int(m.group(2)), **{k: float(vals[k]) for k in KEYS if k in vals}}
            target = next((r for r in runs if m.group(1).startswith(r["config"].get("name", "\0"))), None)
            if target is None:
                target = {"config": {"name": m.group(1)}, "evaluations": []}
                runs.append(target)
            if ev not in target["evaluations"]:
                target["evaluations"].append(ev)
        errors = sorted(set(re.findall(r"(\w+Error: [^\n]{0,120})", re.sub(r"\x1b\[[0-9;]*m", "", text))))
        rec = {"notebook": nb_path.name, "has_outputs": bool(text.strip()), "runs": runs, "errors": errors}
        (OUT / (nb_path.stem + ".json")).write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        for r in runs:
            for ev in r["evaluations"]:
                table.append({"notebook": nb_path.name, **{k: r["config"].get(k, "") for k in ARGS}, **ev})
        if not any(r["evaluations"] for r in runs):
            table.append({"notebook": nb_path.name, "name": "(no retrieval result in saved outputs)",
                          "run": "; ".join(errors)[:200]})
    cols = ["notebook", *ARGS, "run", "epoch", *KEYS]
    with open(OUT / "legacy_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(table)
    for row in table:
        print(row["notebook"], row.get("name"), row.get("mean-recall", ""))


if __name__ == "__main__":
    main()
