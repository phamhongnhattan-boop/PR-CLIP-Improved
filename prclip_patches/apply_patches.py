"""Apply the PR-CLIP-Improved patches to the authors' code (ADMIS-TONGJI/PR-CLIP @ 242fba6).

Usage (inside Colab, from the authors' repo root):
    python /path/to/PR-CLIP-Improved/prclip_patches/apply_patches.py .

What it does (idempotent, every replacement must match exactly once):
  1. Compatibility fixes for current timm / numpy / open_clip 2.20 (same as the team notebooks).
  2. Lightweight checkpoints (best.pt fp16, epoch_latest.pt removed after training).
  3. A3 hooks: RIP-fixed, E1, EXT v2. They are OFF unless the env flags are set
     (PRCLIP_RIP / PRCLIP_E1 / PRCLIP_EXT2), so B1 runs the original behaviour.
"""
import py_compile
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
HERE = Path(__file__).resolve().parent


def read(p):
    return (ROOT / p).read_text(encoding="utf-8").replace("\r\n", "\n")


def write(p, s):
    (ROOT / p).write_text(s, encoding="utf-8")


def once(s, old, new, name):
    n = s.count(old)
    if n != 1:
        raise RuntimeError(f"[{name}] expected 1 match, found {n}")
    return s.replace(old, new)


def soft(s, old, new):
    return s.replace(old, new) if old in s else s


# ---------------------------------------------------------------- 1. compat
s = read("training/optimization.py")
s = soft(s, "from timm.optim.nadam import Nadam\n", "from timm.optim.nadam import NAdamLegacy as Nadam\n")
s = soft(s, "from timm.optim.radam import RAdam\n", "from timm.optim.radam import RAdamLegacy as RAdam\n")
write("training/optimization.py", s)
for p in ("data/train_data.py", "utils/tsne_checkpoints.py"):
    write(p, read(p).replace("np.string_", "np.bytes_"))
write("evaluation/linear_eval.py", read("evaluation/linear_eval.py").replace("astype(np.float)", "astype(float)"))

s = read("training/params.py")
s = re.sub(r'("--load-clip-weights",\s*\n?\s*action="store_true",\s*\n?\s*default=)False', r"\1True", s)
write("training/params.py", s)

s = read("training/main.py")
if "weights_only=False" not in s:
    s = soft(s, "torch.load(args.resume, map_location=device)",
             "torch.load(args.resume, map_location=device, weights_only=False)")
if "[A3]" not in s:
    s = once(s, "from model.model import get_model\n",
             "from model.model import get_model\nfrom model.a3_modules import active_flags as _a3_flags\n", "main-import")
    s = once(s, "    if args.eval_first:",
             "    logging.info(f\"[A3] active modules: {_a3_flags()}\")\n    if args.eval_first:", "main-log")
write("training/main.py", s)

# optional dependencies: import only when the corresponding builder is used, so a
# minimal env (torch + open_clip + transformers + timm) is enough for ViT-B/32 openclip runs
s = read("model/model.py")
if "[A3-LAZY]" not in s:
    s = soft(s, "\nimport cn_clip.clip as cn_clip\n", "\n# [A3-LAZY] optional imports moved into their branches\n")
    s = soft(s, "\nimport adapters\nfrom sentence_transformers import SentenceTransformer\n", "\n")
    s = soft(s, "        CLIP_model, preprocess_val = cn_clip.load_from_name(",
             "        import cn_clip.clip as cn_clip\n        CLIP_model, preprocess_val = cn_clip.load_from_name(")
    s = soft(s, "            if args.adapter=='bottleneck_adapter':",
             "            import adapters\n            if args.adapter=='bottleneck_adapter':")
    s = soft(s, "        text_backbone = SentenceTransformer(",
             "        from sentence_transformers import SentenceTransformer\n        text_backbone = SentenceTransformer(")
write("model/model.py", s)
s = read("training/main.py")
if "except ImportError:\n    wandb = None" not in s:
    s = soft(s, "\nimport wandb\n", "\ntry:\n    import wandb\nexcept ImportError:\n    wandb = None\n")
write("training/main.py", s)

# ---------------------------------------------------------------- 2. model.py
shutil.copy(HERE / "a3_modules.py", ROOT / "model" / "a3_modules.py")
s = read("model/model.py")
if "[A3-PATCH]" not in s:
    s = once(s, "from model.xbert import BertModel\n",
             "from model.xbert import BertModel\nfrom model import a3_modules as A3  # [A3-PATCH]\n", "import")

    # open_clip 2.20 has no return_hidden: compute ViT tokens explicitly (768-d, before proj)
    helper = '''    def _vit_hidden(self, images):
        v = self.image_backbone
        x = v.conv1(images.type(v.conv1.weight.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = v.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        x = x + v.positional_embedding.to(x.dtype)
        x = v.patch_dropout(x)
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = v.transformer(x)
        x = x.permute(1, 0, 2)
        return v.ln_post(x)

'''
    s = once(s, "    def forward(self, images, texts, text_only=False):",
             helper + "    def forward(self, images, texts, text_only=False):", "vit-hidden")
    s = once(s, "self.image_backbone(images, return_hidden=True)", "self._vit_hidden(images)", "return-hidden")

    # store text tokens validity mask (used by E1)
    s = once(s, "                texts = self.tokenizer(texts, context_length=context_length).to(self.device)\n",
             "                texts = self.tokenizer(texts, context_length=context_length).to(self.device)\n"
             "                self.text_valid_mask = A3.text_valid_mask(texts) if self.prompt is None else None\n",
             "token-mask")
    s = once(s, "        self.text_hidden = torch.tensor(1)\n",
             "        self.text_hidden = torch.tensor(1)\n"
             "        self.text_valid_mask = None\n"
             "        self.a3_rip = A3.flag('PRCLIP_RIP')\n"
             "        self.a3_e1 = A3.flag('PRCLIP_E1')\n"
             "        self.a3_ext2 = A3.flag('PRCLIP_EXT2')\n"
             "        if self.a3_e1:\n"
             "            A3.clip_init_projections(self.proj_image, self.proj_text, self.image_backbone, self.text_backbone)\n",
             "init-flags")

    # RIP-fixed inside encode_image (used by training AND retrieval evaluation)
    s = once(s, "            image_features = self.image_backbone(images)\n            if 'vicregl' in self.image_model_tag:",
             "            if getattr(self, 'a3_rip', False) and self.image_model_builder == 'openclip':\n"
             "                image_features = A3.rip_pool(self.image_backbone, images)\n"
             "            else:\n"
             "                image_features = self.image_backbone(images)\n"
             "            if 'vicregl' in self.image_model_tag:",
             "rip")

    # extraction: E1 padding mask / EXT v2 cross-only attention
    s = once(s, "        att_embed = self.it_encoder(it_seq)\n",
             "        att_embed = A3.run_extraction(self.it_encoder, image_embed, text_embed,\n"
             "                                      self.text_valid_mask, self.a3_e1, self.a3_ext2)\n",
             "extraction")
    write("model/model.py", s)

# AMP-FIX: the released AMP branch never calls optimizer.zero_grad(), so gradients
# accumulate until they overflow; every step then has grad = nan and GradScaler skips it,
# i.e. the model is never updated. Accumulate over step_lou (=4) batches as in the fp32
# branch, then step and zero the gradients (same fix as the team notebooks).
s = read("training/train.py")
if "# AMP-FIX" not in s:
    old_amp = (
        "            if scaler is not None:\n"
        "                scaler.scale(total_loss).backward()\n"
        "                scaler.unscale_(optimizer)\n"
        "                norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)\n"
        "                if args.horovod:\n"
        "                    optimizer.synchronize()\n"
        "                    scaler.unscale_(optimizer)\n"
        "                    with optimizer.skip_synchronize():\n"
        "                        scaler.step(optimizer)\n"
        "                else:\n"
        "                    scaler.step(optimizer)\n"
        "                scaler.update()\n")
    new_amp = (
        "            if scaler is not None:  # AMP-FIX\n"
        "                scaler.scale(total_loss).backward()\n"
        "                if (i + 1) % step_lou == 0 or i == train_loader_size - 1:\n"
        "                    scaler.unscale_(optimizer)\n"
        "                    norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)\n"
        "                    if args.horovod:\n"
        "                        optimizer.synchronize()\n"
        "                        with optimizer.skip_synchronize():\n"
        "                            scaler.step(optimizer)\n"
        "                    else:\n"
        "                        scaler.step(optimizer)\n"
        "                    scaler.update()\n"
        "                    optimizer.zero_grad()\n")
    s = once(s, old_amp, new_amp, "amp-fix")
    s = once(s, "    train_loader_size = len(dataloader)\n    step_lou = 4\n",
             "    train_loader_size = len(dataloader)\n    step_lou = 4\n    norm = torch.tensor(0.0)\n", "amp-norm")
    write("training/train.py", s)

# lazy optional evaluations (same as team fork phamhongnhattan-boop/PR-CLIP@learn-cmc)
s = read("evaluation/evaluation.py")
if "[A3-LAZY-EVAL]" not in s:
    s = soft(s, "from .linear_eval import linear_eval\nfrom .zero_shot import zero_shot_eval\n", "# [A3-LAZY-EVAL]\n")
    s = soft(s, "from .nlp_evaluations import nlp_eval\n", "")
    s = soft(s, "    nlp_metrics = nlp_eval(model, epoch, args)\n",
             "    if args.nlp_eval_frequency:\n        from .nlp_evaluations import nlp_eval\n"
             "        nlp_metrics = nlp_eval(model, epoch, args)\n    else:\n        nlp_metrics = {}\n")
    s = soft(s, "    for zeroshot_dataset in zeroshot_datasets:\n",
             "    if args.zeroshot_frequency:\n        from .zero_shot import zero_shot_eval\n    for zeroshot_dataset in (zeroshot_datasets if args.zeroshot_frequency else []):\n")
    s = soft(s, "    if linear_eval_datasets:\n", "    if linear_eval_datasets and args.linear_frequency:\n        from .linear_eval import linear_eval\n")
    write("evaluation/evaluation.py", s)
s = read("__init__.py")
if "def get_model" not in s:
    write("__init__.py", "def get_model(*args, **kwargs):\n    from model.model import get_model as _g\n    return _g(*args, **kwargs)\n")

# ---------------------------------------------------------------- 3. metrics.jsonl
# Every retrieval evaluation appends one JSON line (epoch, csv, flags, metrics,
# wall-clock inference time IT(s)) to <logs>/<name>/metrics.jsonl.
s = read("evaluation/evaluation.py")
if "[A3-METRICS]" not in s:
    s = once(s, "    retrieval_metrics = retrieval_evaluation(model, epoch, preprocess, args)\n",
             "    import time as _t, json as _j, os as _o  # [A3-METRICS]\n"
             "    from model.a3_modules import active_flags as _flags\n"
             "    if torch.cuda.is_available():\n"
             "        torch.cuda.synchronize()\n"
             "    _t0 = _t.time()\n"
             "    retrieval_metrics = retrieval_evaluation(model, epoch, preprocess, args)\n"
             "    if torch.cuda.is_available():\n"
             "        torch.cuda.synchronize()\n"
             "    _it = _t.time() - _t0\n"
             "    if getattr(args, 'logs', None) and str(args.logs).lower() != 'none':\n"
             "        _p = _o.path.join(args.logs, args.name, 'metrics.jsonl')\n"
             "        _o.makedirs(_o.path.dirname(_p), exist_ok=True)\n"
             "        with open(_p, 'a') as _f:\n"
             "            _f.write(_j.dumps({'name': args.name, 'epoch': epoch, 'seed': args.seed,\n"
             "                               'retrieval_data': args.retrieval_data, 'modules': _flags(),\n"
             "                               'resume': args.resume, 'inference_time_s': round(_it, 2),\n"
             "                               **{k: round(float(v), 4) for k, v in retrieval_metrics.items()}}) + '\\n')\n",
             "metrics")
    if "\nimport torch\n" not in s:
        s = "import torch\n" + s
    write("evaluation/evaluation.py", s)

for p in ("model/model.py", "model/a3_modules.py", "training/main.py", "training/optimization.py",
          "evaluation/evaluation.py", "training/train.py"):
    py_compile.compile(str(ROOT / p), doraise=True)
print("Patches applied and compiled OK:", ROOT)
