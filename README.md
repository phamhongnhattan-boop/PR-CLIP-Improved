# PR-CLIP-Improved

Empirical study of parameter-free modifications to **PR-CLIP** ([Guan et al., *Remote Sensing* 2025](https://doi.org/10.3390/rs17132117)) for remote sensing image–text retrieval on RSICD and RSITMD.

- **RIP-fixed**: D4 rotation-invariant pooling of image embeddings (8 rotations/flips), used in training and retrieval.
- **E1**: text padding mask in the extraction encoder + CLIP-initialised extraction projections.
- **EXT v2**: extraction encoder restricted to cross-modal-only attention.

None of the three adds learned parameters. All code is a patch on the authors' code ([ADMIS-TONGJI/PR-CLIP](https://github.com/ADMIS-TONGJI/PR-CLIP), commit `242fba6`). Report: [`report/main.pdf`](report/main.pdf).

## Repository layout

| Path | Content |
|---|---|
| `prclip_patches/a3_modules.py` | RIP-fixed, E1, EXT v2 (switched by env vars) |
| `prclip_patches/apply_patches.py` | Applies compatibility fixes + A3 hooks + `metrics.jsonl` logging to the authors' repo |
| `prclip_patches/prclip_improved_242fba6.patch` | Same changes as a `git diff` against commit `242fba6` |
| `scripts/make_val_split.py` | Image-level validation split from RET-2 (+ SHA-256 manifest) |
| `scripts/run_a3.py` | Runs one A3 configuration (`eval` or `train`) |
| `scripts/aggregate_results.py` | mean ± std tables → `results/a3/`, `report/tables/` |
| `scripts/eval_epoch0_standalone.py` | Epoch-0 evaluation (B1 / RIP-fixed) with only torch + open_clip; matches the authors' recall code |
| `scripts/bootstrap_epoch0.py` | Paired bootstrap CI of the RIP-fixed gain |
| `scripts/RUN_A3_GPU.bat` / `RUN_A3_GPU.ps1` | One-click A3 run on a Windows GPU with an existing env (venv + deps) |
| `scripts/run_a3_local.ps1` | Same A3 protocol on a local Windows GPU (PowerShell) |
| `scripts/extract_notebook_logs.py` | Pulls metrics from saved notebook outputs → `results/legacy/` |
| `tests/test_a3_modules.py` | CPU unit tests (D4 invariance, norm, gradients, masks) |
| `notebooks/` | Colab notebooks (see table below) |
| `results/legacy/` | Metrics extracted from notebook outputs |
| `results/a3/` | Multi-seed A3 results (filled by `A3_MultiSeed_Colab.ipynb`) |
| `results/objective_ablation/` | Loss-objective ablation from OpenAI CLIP (paper PRC vs released MSE vs CMC-only), copied from the team fork |
| `report/` | LaTeX source (IEEEtran) and PDF |

## Environment

Google Colab with a GPU (T4 or better). Packages, as in the notebooks:

```bash
pip install open_clip_torch==2.20.0 clip-benchmark==1.6.1 transformers==4.45.2 adapters timm \
            sentence-transformers wandb pyyaml gdown pytest "protobuf>=6.31.1,<7"
pip install cn_clip --no-deps
pip install faiss-cpu click
```

## Data and checkpoint

```bash
git clone https://github.com/ADMIS-TONGJI/PR-CLIP && cd PR-CLIP && git checkout 242fba67
python /path/to/PR-CLIP-Improved/prclip_patches/apply_patches.py .

# RSICD + RSITMD images (archive used by all team notebooks)
gdown --fuzzy "https://drive.google.com/file/d/15htMe2iE4kVujjWURxfCxZXk8vAAetoT/view" -O images.zip
mkdir images && tar -xzf images.zip -C images && find images -type f -exec mv -t images {} +

# Starting checkpoint: GeoRSCLIP RS5M ViT-B/32 fine-tuned on RET-2 (~605 MB)
mkdir -p cache/weights
wget -O cache/weights/RET-2.pt https://huggingface.co/Zilun/GeoRSCLIP/resolve/main/ckpt/RS5M_ViT-B-32_RET-2.pt

# Validation split (needed because rsicd_val.csv == rsicd_test.csv in the released CSVs)
python /path/to/PR-CLIP-Improved/scripts/make_val_split.py --repo . --val-frac 0.1 --seed 0
```

CSV files (`dataset/train/ret2.csv`, `dataset/test/*.csv`) come with the authors' repo.

> **Split warning.** In the authors' repo `dataset/test/rsicd_val.csv` is byte-identical to `rsicd_test.csv` (SHA-256 `babb5001…`), and 733 of 856 `rsitmd_val.csv` images are also in `ret2.csv`. Do not use either file for model selection.

## Running the A3 study

Open `notebooks/A3_MultiSeed_Colab.ipynb` in Colab and run all cells. It is resumable across sessions (finished runs are skipped). On a local Windows GPU use `scripts\run_a3_local.ps1 -PrClip <authors' repo> -Images <image folder>`. Or step by step from the patched authors' repo:

```bash
# epoch 0, no training (deterministic)
python scripts/run_a3.py eval  --cfg B1 --data rsicd_test --repo /content/PR-CLIP --logs LOGS
python scripts/run_a3.py eval  --cfg R2 --data rsicd_test --repo /content/PR-CLIP --logs LOGS
# one epoch, seeds 1-3, val before/after training
python scripts/run_a3.py train --cfg R3 --seed 2 --repo /content/PR-CLIP --logs LOGS
python scripts/run_a3.py eval  --cfg R3 --seed 2 --data rsicd_test --resume LOGS/A3_R3_train_s2*/checkpoints/epoch_1.pt ...
# tables
python scripts/aggregate_results.py --logs LOGS --out .
```

| Config | `PRCLIP_RIP` | `PRCLIP_E1` | `PRCLIP_EXT2` |
|---|---|---|---|
| B1 baseline | 0 | 0 | 0 |
| R2 | 1 | 0 | 0 |
| R4 | 1 | 1 | 0 |
| R3 | 1 | 1 | 1 |

Shared setup: 1 epoch, batch 16, lr 2e-7, warm-up 100, weight decay 0.5, max grad norm 50, AMP, ViT-B/32, RET-2 init. Every evaluation appends a line to `LOGS/<run>/metrics.jsonl` with all Recall values, mR and inference time IT(s).

Unit tests (CPU): `python -m pytest -q tests`

## Results obtained so far

Epoch 0 (RET-2 checkpoint, no training; deterministic, two identical runs; `results/a3/epoch0_*`):

| Test set | B1 mR | RIP-fixed mR | Gain (95% paired bootstrap CI) | PR-CLIP (paper) |
|---|---|---|---|---|
| RSICD | 38.86 | **39.70** | +0.84 [+0.29, +1.42] | 39.23 |
| RSITMD | 51.81 | 51.88 | +0.07 [−0.77, +0.87] | 52.41 |

B1 on RSITMD reproduces the GeoRSCLIP row of the PR-CLIP paper exactly (all six recalls). Image encoding on 2 CPU threads: RSICD 50 s vs 402 s (8 passes).

```bash
python scripts/eval_epoch0_standalone.py --ckpt RET-2.pt --csv dataset/test/rsicd_test.csv --images images \
       --out results/a3/epoch0_rsicd_test.json --save-emb emb_rsicd.pt
python scripts/bootstrap_epoch0.py --emb emb_rsicd.pt --out results/a3/bootstrap_epoch0_rsicd_test.json
```

The three-seed trained runs (B1/R2/R3) still need a GPU: run `notebooks/A3_MultiSeed_Colab.ipynb`.

## Notebooks and results

| Notebook | What it runs | Result in saved outputs | Report |
|---|---|---|---|
| `A3_MultiSeed_Colab.ipynb` | A3 protocol: epoch 0 (B1, RIP), B1/R2/R3 × 3 seeds, val selection | epoch 0 done (above); trained runs to be run | Tables IV, V |
| `PR_CLIP_(2).ipynb` | Modified extraction (type embeddings, pre-LN, GELU, padding mask) | mR **38.74** | Table VI |
| `PR_CLIP_v4_BRCR.ipynb` | BRCR, flags `mask,bidir,rel,detach` | mR **38.68** | Table VI |
| `PR_CLIP_STPR_Improved_Ready.ipynb` | STPR: 15% masking of reconstruction queries + local cosine loss (β=0.05) | mR 38.69 | not in report |
| `PR_CLIP_AGR_v2_OneEpoch.ipynb` | AGR: learned gated pooling in reconstruction, reconstruction lr 2e-6 | mR 38.59 | not in report |
| `PR_CLIP_v3_DualResidualAdapter_OneEpoch.ipynb` | Dual residual adapter on frozen cached features | failed: `GPU runtime required` | – |
| `PR_CLIP_v3_FIXED_EndToEnd_Train.ipynb` | Dual residual adapter, end-to-end | no metric printed | Table VI (log pending) |
| `PR_CLIP_v4_OneEpoch.ipynb` | BRCR one-epoch template | not executed | – |

All values: RSICD test, 1 epoch, batch 16, lr 2e-7, RET-2 init, seed 1. Full numbers in `results/legacy/legacy_runs.csv`.

**Logs still missing from the repo** (marked † in the report): the earlier sensitivity runs (report Table V), the dual adapter result (38.69), BRCR with all switches (37.97), and the preliminary trained A3 runs (38.72 / 39.09 / 39.10 / 39.11). The epoch-0 numbers are now re-measured with logs. Upload their `out.log` files to `results/raw_logs/<run_name>/`, or replace them with the A3 notebook outputs.

## Related code: paper-objective re-implementation

The team fork [phamhongnhattan-boop/PR-CLIP](https://github.com/phamhongnhattan-boop/PR-CLIP) (branch `learn-cmc`) contains:
- `model/paper_objective.py`: token-level PRC loss and `0.01 × CMC + PRC` (paper Eq. 13), selectable with `--prclip-objective paper`;
- `model/openclip_hidden.py`: ViT-B/32 token path for OpenCLIP 2.20 (same role as `_vit_hidden` in our patch);
- `model/position_extraction.py`: extraction with an optional text padding mask (the masking part of E1);
- `repro/`: an independent re-implementation of Eqs. (3)–(13), dataset audit, and the objective ablation in report Table VIII;
- `REPRODUCTION.md`: paper vs. code differences (report Table I).

## Citation

```bibtex
@article{guan2025prclip,
  title   = {PR-CLIP: Cross-Modal Positional Reconstruction for Remote Sensing Image--Text Retrieval},
  author  = {Guan, Jihong and Shu, Yulou and Li, Wengen and Song, Zihan and Zhang, Yichao},
  journal = {Remote Sensing}, volume = {17}, number = {13}, pages = {2117}, year = {2025},
  doi     = {10.3390/rs17132117}
}
```
