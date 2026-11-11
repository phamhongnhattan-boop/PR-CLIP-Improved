"""CPU unit tests for prclip_patches/a3_modules.py.  Run: python -m pytest -q tests"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prclip_patches"))
import a3_modules as A3  # noqa: E402


class TinyEncoder(nn.Module):
    """Position-sensitive toy image encoder (not rotation invariant)."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, 16)

    def forward(self, x):
        return self.fc(x.flatten(1))


def test_d4_has_eight_distinct_elements_identity_first():
    x = torch.randn(1, 3, 8, 8)
    outs = list(A3.d4_transforms(x))
    assert len(outs) == 8
    assert torch.equal(outs[0], x)
    flat = torch.stack([o.flatten() for o in outs])
    assert torch.cdist(flat, flat).fill_diagonal_(1.0).min() > 0


def test_rip_direction_is_rotation_invariant():
    torch.manual_seed(0)
    enc, x = TinyEncoder(), torch.randn(2, 3, 8, 8)
    a = F.normalize(A3.rip_pool(enc, x), dim=-1)
    for g in list(A3.d4_transforms(x))[1:]:
        b = F.normalize(A3.rip_pool(enc, g), dim=-1)
        assert torch.allclose(a, b, atol=1e-5)


def test_rip_keeps_norm_of_identity_branch():
    torch.manual_seed(0)
    enc, x = TinyEncoder(), torch.randn(2, 3, 8, 8)
    assert torch.allclose(A3.rip_pool(enc, x).norm(dim=-1), enc(x).norm(dim=-1), atol=1e-5)


def test_rip_gradient_only_from_identity_branch():
    torch.manual_seed(0)
    enc, x = TinyEncoder(), torch.randn(2, 3, 8, 8)
    out = A3.rip_pool(enc, x).sum()
    out.backward()
    assert enc.fc.weight.grad is not None and torch.isfinite(enc.fc.weight.grad).all()


def test_cross_only_mask_blocks_same_modality():
    m = A3.cross_only_attn_mask(3, 2, "cpu")
    assert m[:3, :3].all() and m[3:, 3:].all()
    assert not m[:3, 3:].any() and not m[3:, :3].any()


def test_text_valid_mask_stops_at_eot():
    tokens = torch.tensor([[49406, 320, 1125, 49407, 0, 0]])  # SOT a cat EOT pad pad
    assert A3.text_valid_mask(tokens).tolist() == [[True, True, True, True, False, False]]


def _encoder():
    layer = nn.TransformerEncoderLayer(16, 4, 32, 0.0, batch_first=True)
    return nn.TransformerEncoder(layer, 2, enable_nested_tensor=False).eval()


def test_extraction_baseline_matches_original_call():
    torch.manual_seed(0)
    enc, img, txt = _encoder(), torch.randn(2, 5, 16), torch.randn(2, 7, 16)
    ref = enc(torch.cat([img, txt], 1))
    out = A3.run_extraction(enc, img, txt, None, use_e1=False, use_ext2=False)
    assert torch.allclose(ref, out, atol=1e-6)


def test_extraction_variants_finite_and_mask_effective():
    torch.manual_seed(0)
    enc, img, txt = _encoder(), torch.randn(2, 5, 16), torch.randn(2, 7, 16)
    tmask = torch.tensor([[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    for e1, ext2 in [(True, False), (False, True), (True, True)]:
        out = A3.run_extraction(enc, img, txt, tmask, use_e1=e1, use_ext2=ext2)
        assert out.shape == (2, 12, 16) and torch.isfinite(out).all()
    # with E1, changing padded text tokens must not change image outputs
    a = A3.run_extraction(enc, img, txt, tmask, use_e1=True, use_ext2=True)
    txt2 = txt.clone()
    txt2[0, 3:] += 10.0
    b = A3.run_extraction(enc, img, txt2, tmask, use_e1=True, use_ext2=True)
    assert torch.allclose(a[0, :5], b[0, :5], atol=1e-5)


def test_clip_init_projections_copies_weights_without_new_params():
    pi, pt = nn.Linear(8, 4), nn.Linear(4, 4)
    vis = nn.Module(); vis.proj = nn.Parameter(torch.randn(8, 4))
    txt = nn.Module(); txt.text_projection = nn.Parameter(torch.randn(4, 4))
    n_before = sum(p.numel() for p in pi.parameters()) + sum(p.numel() for p in pt.parameters())
    A3.clip_init_projections(pi, pt, vis, txt)
    x = torch.randn(3, 8)
    assert torch.allclose(pi(x), x @ vis.proj, atol=1e-5)
    assert n_before == sum(p.numel() for p in pi.parameters()) + sum(p.numel() for p in pt.parameters())
