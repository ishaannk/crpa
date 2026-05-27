"""
evaluate.py — Evaluation functions reproducing Tables 2–5 and figures.

Functions:
  retrieval_accuracy    — Needle-in-Haystack top-1 accuracy
  retrieval_by_depth    — accuracy vs needle position (depth sweep)
  measure_overlap       — average Jaccard overlap using top-p support
  measure_throughput    — runtime in ms per forward pass
  routing_diagnostics   — empty partition %, routing entropy, load error
"""

import math
import time
import random
import torch
import torch.nn.functional as F

from config import CFG
from data import get_lm_batch, make_needle_batch


# ─────────────────────────────────────────────────────────────────────────────
#  Table 2  — Retrieval Accuracy
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def retrieval_accuracy(model, block_size, device,
                       n_batches=30, bs=8):
    """
    Top-1 accuracy on Needle-in-Haystack.
    Model must predict the value token at the final (query) position.
    """
    model.eval()
    correct = total = 0
    for _ in range(n_batches):
        x, y      = make_needle_batch(block_size, bs=bs, device=device)
        logits, _ = model(x)
        pred      = logits[:, -1, :].argmax(dim=-1)
        correct  += (pred == y).sum().item()
        total    += y.shape[0]
    model.train()
    return 100.0 * correct / total


# ─────────────────────────────────────────────────────────────────────────────
#  Cell 10  — Needle Depth Sweep (key reviewer question)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def retrieval_by_depth(model, block_size, depth, device, n_batches=20, bs=8):
    """
    Retrieval accuracy when needle is placed at a specific sequence depth.
    Sliding window fails at deep positions (needle outside window).
    CRPA-causal should maintain accuracy across all depths.
    """
    model.eval()
    correct = total = 0
    for _ in range(n_batches):
        x, y      = make_needle_batch(block_size, bs=bs,
                                       needle_depth=depth, device=device)
        logits, _ = model(x)
        pred      = logits[:, -1, :].argmax(dim=-1)
        correct  += (pred == y).sum().item()
        total    += y.shape[0]
    model.train()
    return 100.0 * correct / total


# ─────────────────────────────────────────────────────────────────────────────
#  Table 4  — Overlap Measurement
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def measure_overlap(model, block_size, device, n_batches=12):
    """
    Average Jaccard overlap using top-p support (within-partition sampling).

    Why within-partition:
      Cross-partition pairs share only relay tokens (low Jaccard by design).
      Meaningful overlap only exists within the same partition window.
    """
    model.eval()
    ovs = []
    for _ in range(n_batches):
        x, _ = get_lm_batch('val', block_size, bs=4, device=device)
        model(x)
        for blk in model.blocks:
            A = blk.attn._Alast
            if A is None:
                continue
            Am = A.mean(dim=(0, 1))   # (T, T)
            T  = Am.shape[0]
            for p in range(math.ceil(T / blk.attn.p_size)):
                s = p * blk.attn.p_size
                e = min(s + blk.attn.p_size, T)
                if e - s < 2:
                    continue
                for _ in range(8):
                    i = random.randint(s, e-1)
                    j = random.randint(s, e-1)
                    if i != j:
                        ovs.append(blk.attn._jaccard(Am, i, j))
    model.train()
    import numpy as np
    return float(np.mean(ovs)) if ovs else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Table 3  — Runtime Scaling
# ─────────────────────────────────────────────────────────────────────────────

def measure_throughput(model_cls, variant, ctx, vocab_size, device,
                       n_runs=60, bs=8):
    """
    Forward-pass latency in ms.
    Returns (ms_per_run, normalised_factor_vs_baseline).
    """
    import torch
    torch.manual_seed(0)
    m = model_cls(variant, ctx, vocab_size, device).to(device).eval()
    x = torch.randint(0, vocab_size, (bs, ctx), device=device)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            m(x)
    if device == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            m(x)
    if device == 'cuda':
        torch.cuda.synchronize()

    ms = (time.perf_counter() - t0) / n_runs * 1000
    del m
    if device == 'cuda':
        torch.cuda.empty_cache()
    return ms


# ─────────────────────────────────────────────────────────────────────────────
#  Table 5  — Routing Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def routing_diagnostics(model, block_size, device, n_batches=20):
    """
    Measures routing quality:
      empty_frac  — % of partitions with < 1% token utilisation (lower = better)
      routing_ent — entropy of routing distribution (higher = more balanced)
      load_error  — std of partition utilisation (lower = more balanced)
    """
    model.eval()
    all_soft = []
    for _ in range(n_batches):
        x, _ = get_lm_batch('val', block_size, bs=8, device=device)
        emb  = model.drop(
            model.tok_emb(x) +
            model.pos_emb(torch.arange(x.shape[1], device=device)))
        for blk in model.blocks:
            if hasattr(blk.attn, 'router'):
                soft, _ = blk.attn.router(emb)
                all_soft.append(soft.detach().cpu())
    model.train()

    if not all_soft:
        return 0.0, 0.0, 0.0

    import torch as _t
    S   = _t.cat(all_soft).reshape(-1, all_soft[0].shape[-1])
    avg = S.mean(0)

    empty_frac  = (avg < 0.01).float().mean().item() * 100
    routing_ent = -(S * (S + 1e-9).log()).sum(-1).mean().item()
    load_error  = avg.std().item()

    return empty_frac, routing_ent, load_error
