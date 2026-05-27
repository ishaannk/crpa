"""
train.py — Training loop for CRPA and all ablation variants.

Co-training strategy:
  - ret_ratio % of steps: train on Needle-in-Haystack retrieval task
  - (1 - ret_ratio) % of steps: train on WikiText-2 language modelling
  This is required because the model needs to learn retrieval during
  training, not just at evaluation time.

Sensitivity estimation (crpa_causal only):
  - Every sens_interval steps, estimate Delta_ij for high-overlap pairs
  - Uses LM validation batches (shape (bs, block_size))
  - Results cached in each attention layer's _sens dict
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG
from data import get_lm_batch, make_needle_batch


def make_lr_schedule(optimizer, warmup, total):
    """Linear warmup → cosine decay."""
    def fn(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


@torch.no_grad()
def estimate_loss(model, block_size, device):
    """Mean train/val LM loss over eval_iters batches."""
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(CFG['eval_iters'])
        for k in range(CFG['eval_iters']):
            x, y      = get_lm_batch(split, block_size, bs=8, device=device)
            _, loss   = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(model, block_size, device,
          max_iters=None, verbose=True):
    """
    Train one GPT variant.

    Returns:
        model:       trained model (in-place)
        val_hist:    list of validation losses at eval checkpoints
        step_hist:   list of step numbers corresponding to val_hist
    """
    max_iters = max_iters or CFG['max_iters']

    use_amp = (device == 'cuda' or (hasattr(device, 'type') and 'cuda' in str(device)))
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    opt   = torch.optim.AdamW(
        model.parameters(),
        lr           = CFG['lr'],
        weight_decay = CFG['weight_decay'])
    sched = make_lr_schedule(opt, CFG['warmup_steps'], max_iters)

    val_hist, step_hist = [], []

    # Needle validation batch for sensitivity estimation.
    # Needle loss (last-token CE) reveals which overlaps are retrieval-critical:
    # if masking (i,j) doesn't hurt retrieval → Δ_ij ≤ eps → safe to penalize.
    nvx, nvy = make_needle_batch(block_size, bs=8, device=device)

    if verbose:
        print(f"  variant={model.variant} | "
              f"{model.n_params()/1e6:.1f}M params | "
              f"block={block_size}")

    for step in range(max_iters):

        # ── Sensitivity estimation (crpa_causal only, intermittent) ──────────
        if (model.variant == 'crpa_causal'
                and step >= 600
                and step % CFG['sens_interval'] == 0):
            for blk in model.blocks:
                blk.attn.update_sensitivity(model, nvx, nvy)
            # Refresh needle batch so sensitivity tracks current model behaviour
            nvx, nvy = make_needle_batch(block_size, bs=8, device=device)

        # ── Co-training step ──────────────────────────────────────────────────
        with torch.autocast(device_type='cuda' if use_amp else 'cpu',
                            dtype=torch.bfloat16, enabled=use_amp):
            if random.random() < CFG['ret_ratio']:
                x, y_ret  = make_needle_batch(block_size, n_needles=2, device=device)
                logits, _ = model(x)
                loss       = F.cross_entropy(logits[:, -1, :], y_ret)
            else:
                x, y   = get_lm_batch('train', block_size, device=device)
                _, loss = model(x, y)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), CFG['grad_clip'])
        scaler.step(opt)
        scaler.update()
        sched.step()

        # ── Logging ───────────────────────────────────────────────────────────
        if step % CFG['eval_interval'] == 0 or step == max_iters - 1:
            L = estimate_loss(model, block_size, device)
            val_hist.append(L['val'])
            step_hist.append(step)
            if verbose:
                print(f"    step {step:>5}  "
                      f"train={L['train']:.4f}  val={L['val']:.4f}")

    return model, val_hist, step_hist
