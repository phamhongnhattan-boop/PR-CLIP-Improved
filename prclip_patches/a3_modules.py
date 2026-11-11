"""A3 modules for PR-CLIP (authors' code, commit 242fba6).

All modules are switched on/off with environment variables, so the same
patched code base is used for every run (B1, R2, R3, R4):

    PRCLIP_RIP=1   RIP-fixed: D4 rotation-invariant pooling of image embeddings
    PRCLIP_E1=1    E1: text padding mask + CLIP-initialised extraction projections
    PRCLIP_EXT2=1  EXT v2: cross-modal-only attention inside the extraction encoder

None of the modules adds learned parameters or new hyper-parameters.
"""
import os

import torch
import torch.nn.functional as F


def flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def active_flags() -> str:
    names = [n for n in ("PRCLIP_RIP", "PRCLIP_E1", "PRCLIP_EXT2") if flag(n)]
    return ",".join(names) if names else "none (B1 baseline)"


# ---------------------------------------------------------------------------
# RIP-fixed: D4 rotation-invariant pooling
# ---------------------------------------------------------------------------
def d4_transforms(images: torch.Tensor):
    """Yield the 8 elements of the dihedral group D4 applied to a [B,C,H,W] batch.

    The identity comes first. Images must be square (224x224 after CLIP preprocessing).
    """
    assert images.shape[-1] == images.shape[-2], "D4 pooling needs square images"
    for flip in (False, True):
        x = torch.flip(images, dims=[-1]) if flip else images
        for k in range(4):
            yield torch.rot90(x, k=k, dims=[-2, -1]) if k else x


def rip_pool(encode_fn, images: torch.Tensor) -> torch.Tensor:
    """RIP-fixed pooling (report Eq. 6-7).

    v_bar = mean_g normalize(f(g(I)));  v_D4 = normalize(v_bar) * ||f(I)||.
    Only the identity branch keeps gradients; the other 7 run under no_grad.
    """
    feats = None
    base = None
    for idx, x in enumerate(d4_transforms(images)):
        if idx == 0:
            f = encode_fn(x)
            base = f
        else:
            with torch.no_grad():
                f = encode_fn(x)
        f = F.normalize(f.float(), dim=-1)
        feats = f if feats is None else feats + f
    v_bar = feats / 8.0
    return F.normalize(v_bar, dim=-1) * base.float().norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# E1: CLIP-initialised projections + text padding mask
# ---------------------------------------------------------------------------
@torch.no_grad()
def clip_init_projections(proj_image: torch.nn.Linear, proj_text: torch.nn.Linear,
                          image_backbone, text_backbone) -> None:
    """Initialise proj_image (768->512) with CLIP visual.proj and proj_text
    (512->512) with CLIP text_projection, biases set to zero. No new parameters."""
    vproj = getattr(image_backbone, "proj", None)
    if vproj is not None and tuple(vproj.shape) == (proj_image.in_features, proj_image.out_features):
        proj_image.weight.copy_(vproj.t().to(proj_image.weight.dtype))
        proj_image.bias.zero_()
    tproj = getattr(text_backbone, "text_projection", None)
    if tproj is not None and tuple(tproj.shape) == (proj_text.in_features, proj_text.out_features):
        proj_text.weight.copy_(tproj.t().to(proj_text.weight.dtype))
        proj_text.bias.zero_()


def text_valid_mask(tokens: torch.Tensor) -> torch.Tensor:
    """True for real tokens (SOT ... EOT), False for padding after EOT."""
    pos = torch.arange(tokens.size(1), device=tokens.device)[None, :]
    return pos <= tokens.argmax(dim=-1)[:, None]


def key_padding_mask(n_image: int, text_mask: torch.Tensor) -> torch.Tensor:
    """Key padding mask for the [image ; text] sequence. True = ignore."""
    b = text_mask.size(0)
    img = torch.zeros(b, n_image, dtype=torch.bool, device=text_mask.device)
    return torch.cat([img, ~text_mask.bool()], dim=1)


# ---------------------------------------------------------------------------
# EXT v2: cross-modal-only attention, re-using the original 2-layer encoder
# ---------------------------------------------------------------------------
def cross_only_attn_mask(n_image: int, n_text: int, device) -> torch.Tensor:
    """[L, L] boolean mask, True = blocked. Image tokens attend only to text
    tokens and text tokens attend only to image tokens, so the original
    self-attention layers act as bidirectional cross-attention."""
    L = n_image + n_text
    mask = torch.zeros(L, L, dtype=torch.bool, device=device)
    mask[:n_image, :n_image] = True
    mask[n_image:, n_image:] = True
    return mask


def run_extraction(it_encoder, image_embed: torch.Tensor, text_embed: torch.Tensor,
                   text_mask=None, use_e1: bool = False, use_ext2: bool = False) -> torch.Tensor:
    """Drop-in replacement for `self.it_encoder(it_seq)` in ItraModel.forward."""
    n_img, n_txt = image_embed.size(1), text_embed.size(1)
    it_seq = torch.cat([image_embed, text_embed], dim=1)
    kpm = key_padding_mask(n_img, text_mask) if (use_e1 and text_mask is not None) else None
    attn = cross_only_attn_mask(n_img, n_txt, it_seq.device) if use_ext2 else None
    if kpm is None and attn is None:
        return it_encoder(it_seq)
    if kpm is not None and attn is not None:
        # merge into one float mask per sample so every row keeps >=1 key
        neg = torch.finfo(it_seq.dtype).min
        full = attn[None].expand(it_seq.size(0), -1, -1) | kpm[:, None, :]
        nheads = it_encoder.layers[0].self_attn.num_heads
        fmask = torch.zeros(full.shape, dtype=it_seq.dtype, device=it_seq.device).masked_fill(full, neg)
        fmask = fmask.repeat_interleave(nheads, dim=0)
        x = it_seq
        for layer in it_encoder.layers:
            x = layer(x, src_mask=fmask)
        return x
    x = it_seq
    for layer in it_encoder.layers:
        x = layer(x, src_mask=attn, src_key_padding_mask=kpm)
    return x
