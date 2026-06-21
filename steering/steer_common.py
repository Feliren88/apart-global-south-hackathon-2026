"""
Steering engine — shared mechanism for cross-lingual contrastive steering.

This module is the low-level toolkit underneath the 6-phase pipeline in
`run_steering.py`. It deliberately *reuses the inference/ codebase* so that the
steering experiments share the exact same model loader, dataset adapter, MCQ
construction and behavioural taxonomy as the benchmark in ../inference:

    inference.models            LoadedModel / load_model / letter_token_ids
    inference.datasets_adapter  load_records (unified Record schema)
    inference.classify          build_mcq / classify_letter
                                CATEGORIES = image_bias | text_bias |
                                             distractor | conflict_abstain | other

The five behavioural categories map one-to-one onto the methodology's metrics:

    CR (Conflict Rate / abstain)   <- conflict_abstain   (chose option D)
    VR (Image Reliance / visual)   <- image_bias         (faithful to the image)
    TR (Text Override / bias)      <- text_bias          (follows the caption)
    DR (Distractor)                <- distractor
    IR (Incorrect / parse fail)    <- other

What this engine adds on top of inference/:
  * decoder_layers()    — locate the LM decoder layer stack across VLM families
  * forward_score()     — single forward pass: A/B/C/D logit-argmax answer, plus
                          (optional) per-layer last-token activations and a
                          logit-lens abstain-vs-answer trace across depth
  * steer()             — context manager that adds  alpha * v_hat^(k)  to the
                          residual stream at chosen decoder layers via forward
                          hooks (inference-time activation steering)
  * contrastive_vector()— mu_abstain^(k) - mu_assert^(k), L2-normalised
"""

from __future__ import annotations

import os
import re
import sys
from contextlib import contextmanager

import numpy as np
import torch

# ── Reuse the inference/ codebase (single source of truth) ───────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_INFERENCE_DIR = os.path.join(os.path.dirname(_HERE), "inference")
if _INFERENCE_DIR not in sys.path:
    sys.path.insert(0, _INFERENCE_DIR)

import models as mdl            # noqa: E402
import classify as clf          # noqa: E402
import datasets_adapter as dsa  # noqa: E402

# Map a behavioural category -> the methodology's metric code.
CATEGORY_TO_METRIC = {
    "conflict_abstain": "CR",
    "image_bias": "VR",
    "text_bias": "TR",
    "distractor": "DR",
    "other": "IR",
}
METRICS = ["CR", "VR", "TR", "DR", "IR"]


# ── Decoder-layer discovery ──────────────────────────────────────────────────
def decoder_layers(lm: "mdl.LoadedModel") -> list[torch.nn.Module]:
    """Return the ordered list of LM decoder-layer modules of a VLM.

    Robust across families: we collect every sub-module whose class name ends in
    "DecoderLayer", group them by their parent path (the ".<idx>" suffix is the
    layer index), and pick the group whose size matches the LM's reported layer
    count (falling back to the largest group). This skips vision-tower blocks,
    which are conventionally named "...EncoderLayer".
    """
    pat = re.compile(r"^(.*)\.(\d+)$")
    groups: dict[str, list[tuple[int, torch.nn.Module]]] = {}
    for name, module in lm.model.named_modules():
        if not module.__class__.__name__.endswith("DecoderLayer"):
            continue
        m = pat.match(name)
        if not m:
            continue
        prefix, idx = m.group(1), int(m.group(2))
        groups.setdefault(prefix, []).append((idx, module))
    if not groups:
        raise RuntimeError(
            f"No *DecoderLayer modules found in {lm.key}; cannot steer.")

    target_n = lm.num_layers()
    chosen = None
    for items in groups.values():
        items.sort(key=lambda x: x[0])
        if target_n and len(items) == target_n:
            chosen = items
            break
    if chosen is None:
        chosen = sorted(max(groups.values(), key=len), key=lambda x: x[0])
    return [mod for _, mod in chosen]


def target_layer_indices(n_layers: int, frac_lo: float, frac_hi: float) -> list[int]:
    """Decoder-layer indices inside the [frac_lo, frac_hi] depth band."""
    lo = max(0, int(round(frac_lo * n_layers)))
    hi = min(n_layers - 1, int(round(frac_hi * n_layers)))
    if hi < lo:
        hi = lo
    return list(range(lo, hi + 1))


# ── Logit-lens (abstain-vs-answer across depth) ──────────────────────────────
def find_final_norm(model) -> torch.nn.Module | None:
    """Best-effort locate the LM's final RMS/LayerNorm before the unembedding."""
    candidates = [
        "model.norm", "model.language_model.norm", "model.model.norm",
        "language_model.model.norm", "language_model.norm",
        "model.text_model.norm", "thinker.model.norm",
    ]
    for path in candidates:
        obj = model
        ok = True
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok and obj is not None:
            return obj
    return None


class LensTools:
    """Bundle the unembedding + final norm + answer/abstain token ids used for
    the logit-lens 'abstain - answer' trace across decoder depth (Figure 2)."""

    def __init__(self, lm, letter_ids):
        self.head = lm.model.get_output_embeddings()    # lm_head (Linear)
        self.norm = find_final_norm(lm.model)
        self.abstain_ids = letter_ids.get(clf.CONFLICT_LETTER, [])  # D
        self.answer_ids = [i for L in clf.ANSWER_LETTERS              # A/B/C
                           for i in letter_ids.get(L, [])]

    @property
    def usable(self) -> bool:
        return self.head is not None and self.abstain_ids and self.answer_ids

    def diff_trace(self, hidden_states) -> np.ndarray:
        """hidden_states: tuple over (L+1) of [B, seq, hidden]; read the last
        token of row 0. Returns [L+1] of  logit(D) - max logit(A/B/C).

        Vectorised: stack the L+1 last-token vectors and project them in a single
        matmul (one GPU->CPU copy), instead of an L+1-long Python loop.
        """
        with torch.no_grad():
            w = self.head.weight                                  # [vocab, hidden]
            hs = torch.stack([h[0, -1, :] for h in hidden_states],
                             dim=0).to(w.dtype)                   # [L+1, hidden]
            if self.norm is not None:
                try:
                    hs = self.norm(hs)
                except Exception:
                    pass
            logits = torch.matmul(hs, w.t()).float()             # [L+1, vocab]
            d = logits[:, self.abstain_ids].amax(dim=1)          # [L+1]
            a = logits[:, self.answer_ids].amax(dim=1)           # [L+1]
            return (d - a).cpu().numpy().astype(np.float32)

    def diff_trace_batch(self, last: torch.Tensor) -> np.ndarray:
        """Batched logit-lens. `last` is [L+1, B, hidden] (per-row last-token
        hidden at every layer). Returns [B, L+1] of logit(D) - max logit(A/B/C).

        Projects only onto the answer/abstain rows of the unembedding (a few
        token ids), not the full vocab — far cheaper than a [vocab, hidden] matmul.
        """
        with torch.no_grad():
            w = self.head.weight                                 # [vocab, hidden]
            ids = list(self.abstain_ids) + list(self.answer_ids)
            w_sub = w[ids]                                        # [k, hidden]
            hs = last.to(w.dtype)
            if self.norm is not None:
                try:
                    hs = self.norm(hs)
                except Exception:
                    pass
            sub = torch.matmul(hs, w_sub.t()).float()            # [L+1, B, k]
            na = len(self.abstain_ids)
            d = sub[..., :na].amax(dim=-1)                       # [L+1, B]
            a = sub[..., na:].amax(dim=-1)                       # [L+1, B]
            return (d - a).permute(1, 0).cpu().numpy().astype(np.float32)


# ── Scoring (baseline + under steering) ──────────────────────────────────────
def forward_score(lm, inputs, letter_ids, want_hidden=False, lens: "LensTools|None" = None):
    """One forward pass over a single prompt. Returns:
        chosen   : argmax letter over A/B/C/D answer-token logits (or None)
        scores   : {letter: rounded logit margin}
        hidden   : [L+1, hidden] float16 last-token activation per layer, or None
        lens_diff: [L+1] abstain-vs-answer logit-lens trace, or None
    The hidden states / lens trace come from the SAME forward pass (free).
    """
    device = next(lm.model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    need_hs = want_hidden or (lens is not None)
    with torch.no_grad():
        out = lm.model(**inputs, output_hidden_states=need_hs, use_cache=False)

    logits = out.logits[0, -1, :].float()
    scores = {L: max(float(logits[i]) for i in ids)
              for L, ids in letter_ids.items() if ids}
    chosen = max(scores, key=scores.get) if scores else None

    hidden = None
    if want_hidden and getattr(out, "hidden_states", None):
        try:
            # Stack on-device then a single GPU->CPU copy (vs one copy per layer).
            hidden = torch.stack([h[0, -1, :] for h in out.hidden_states], dim=0) \
                .to(torch.float16).cpu().numpy()
        except Exception:
            hidden = None

    lens_diff = None
    if lens is not None and lens.usable and getattr(out, "hidden_states", None):
        try:
            lens_diff = lens.diff_trace(out.hidden_states)
        except Exception:
            lens_diff = None

    return chosen, {L: round(v, 3) for L, v in scores.items()}, hidden, lens_diff


def forward_score_batch(lm, inputs, letter_ids, want_hidden=False,
                        lens: "LensTools|None" = None):
    """Batched counterpart of forward_score over a RIGHT-padded batch.

    `inputs` is the dict from `mdl.build_inputs_batch` (right padded, so each
    row's last real token is at `attention_mask.sum(dim=1) - 1`). A single
    forward scores the whole batch; per-row letter logits, last-token hidden
    states and the logit-lens trace are gathered with one GPU->CPU copy each.

    Returns parallel lists of length B:
        chosen[b]    : argmax letter over A/B/C/D (or None)
        scores[b]    : {letter: rounded logit margin}
        hidden[b]    : [L+1, hidden] float16 per-layer last-token act, or None
        lens_diff[b] : [L+1] abstain-vs-answer trace, or None
    """
    device = next(lm.model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    need_hs = want_hidden or (lens is not None)

    am = inputs.get("attention_mask")
    with torch.no_grad():
        out = lm.model(**inputs, output_hidden_states=need_hs, use_cache=False)

    logits_all = out.logits                                   # [B, seq, vocab]
    B = logits_all.shape[0]
    if am is not None:
        last_idx = am.long().sum(dim=1) - 1                   # [B] last real token
    else:
        last_idx = torch.full((B,), logits_all.shape[1] - 1, device=device)
    bidx = torch.arange(B, device=device)
    logits = logits_all[bidx, last_idx, :].float()           # [B, vocab]

    # Per-letter score = max logit over that letter's token-id variants.
    letters = [L for L, ids in letter_ids.items() if ids]
    per_letter = torch.stack(
        [logits[:, letter_ids[L]].amax(dim=1) for L in letters], dim=1)  # [B, nL]
    best = per_letter.argmax(dim=1).cpu().tolist()
    pl = per_letter.cpu()
    chosen = [letters[b] for b in best] if letters else [None] * B
    scores = [{L: round(float(pl[i, j]), 3) for j, L in enumerate(letters)}
              for i in range(B)]

    hidden = [None] * B
    lens_diff = [None] * B
    if need_hs and getattr(out, "hidden_states", None):
        # Gather only the per-row last token from each layer (avoids stacking the
        # full [L+1, B, seq, hidden] residual stream).
        last = torch.stack([h[bidx, last_idx, :] for h in out.hidden_states],
                           dim=0)                              # [L+1, B, hidden]
        if want_hidden:
            try:
                harr = last.permute(1, 0, 2).to(torch.float16).cpu().numpy()
                hidden = [harr[i] for i in range(B)]
            except Exception:
                hidden = [None] * B
        if lens is not None and lens.usable:
            try:
                larr = lens.diff_trace_batch(last)            # [B, L+1]
                lens_diff = [larr[i] for i in range(B)]
            except Exception:
                lens_diff = [None] * B

    return chosen, scores, hidden, lens_diff


# ── Activation steering ──────────────────────────────────────────────────────
@contextmanager
def steer(layers: list[torch.nn.Module], layer_vecs: dict[int, torch.Tensor],
          alpha: float):
    """Add  alpha * v  to the residual-stream output of each given decoder layer
    for the duration of the context. `layer_vecs` maps decoder-layer index ->
    a [hidden] unit vector already on the model's device. alpha == 0 is a no-op.
    """
    if alpha == 0.0 or not layer_vecs:
        yield
        return

    handles = []

    def make_hook(vec):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                h = out[0]
                h = h + alpha * vec.to(h.dtype)
                return (h,) + tuple(out[1:])
            return out + alpha * vec.to(out.dtype)
        return hook

    try:
        for idx, vec in layer_vecs.items():
            handles.append(layers[idx].register_forward_hook(make_hook(vec)))
        yield
    finally:
        for h in handles:
            h.remove()


def contrastive_vector(acts_abstain: np.ndarray, acts_assert: np.ndarray,
                       layer_col: int) -> np.ndarray | None:
    """v_hat^(k) = normalise( mean(abstain) - mean(assert) ) at hidden-state
    column `layer_col`. `acts_*` are [n, L+1, hidden]. Returns [hidden] float32
    unit vector, or None if either class is empty."""
    if len(acts_abstain) == 0 or len(acts_assert) == 0:
        return None
    mu_a = acts_abstain[:, layer_col, :].astype(np.float32).mean(axis=0)
    mu_f = acts_assert[:, layer_col, :].astype(np.float32).mean(axis=0)
    delta = mu_a - mu_f
    norm = np.linalg.norm(delta)
    if norm < 1e-8:
        return None
    return (delta / norm).astype(np.float32)


def bootstrap_rates(categories: list[str], n_boot: int = 1000,
                    seed: int = 0) -> dict[str, tuple[float, float]]:
    """Per-metric (mean, std) over bootstrap resamples of a category list.
    Returns {metric_code: (rate, std)} for CR/VR/TR/DR/IR."""
    cats = np.asarray([CATEGORY_TO_METRIC.get(c, "IR") for c in categories])
    n = len(cats)
    if n == 0:
        return {m: (0.0, 0.0) for m in METRICS}
    point = {m: float((cats == m).mean()) for m in METRICS}
    rng = np.random.default_rng(seed)
    boot = {m: [] for m in METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = cats[idx]
        for m in METRICS:
            boot[m].append(float((s == m).mean()))
    return {m: (point[m], float(np.std(boot[m]))) for m in METRICS}
