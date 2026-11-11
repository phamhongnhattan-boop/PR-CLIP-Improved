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
          "evaluation/evaluation.py"):
    py_compile.compile(str(ROOT / p), doraise=True)
print("Patches applied and compiled OK:", ROOT)
