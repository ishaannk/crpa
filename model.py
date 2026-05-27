"""
model.py — CRPA Attention and GPT model.

Implements all components from §4 of the paper:
  §4.1  Causally-Regularized Partitioned Attention (CRPA)
  §4.2  Differentiable Routing
  §4.3  Load Balance Loss
  §4.4  Causal Redundancy Regularization
  §4.5  Relay Token Communication
  §4.6  Complexity Analysis (O(n(w+g+k)) per token)

Variants:
  dense        — standard full causal attention (GPT baseline)
  sliding      — sliding-window local attention (Longformer-style)
  crpa_noreg   — CRPA sparse attention, no overlap regularisation
  crpa_naive   — CRPA + penalise all high-overlap pairs (naive ablation)
  crpa_causal  — CRPA + penalise only low-sensitivity pairs (paper claim)
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CFG


# ─────────────────────────────────────────────────────────────────────────────
#  §4.2  Differentiable Router
# ─────────────────────────────────────────────────────────────────────────────

class DifferentiableRouter(nn.Module):
    """
    Maps each token to one of m partitions.
    Soft assignment during training (differentiable).
    Hard argmax at inference.
    """
    def __init__(self, d_model, n_partitions, temp=0.7):
        super().__init__()
        self.T = temp
        self.n = n_partitions
        r_dim  = max(d_model // 8, 16)
        self.Wr    = nn.Linear(d_model, r_dim, bias=False)
        self.cents = nn.Parameter(0.02 * torch.randn(n_partitions, r_dim))

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            soft: (B, T, n_partitions)  soft assignment probabilities
            hard: (B, T)                hard partition index
        """
        r    = self.Wr(x)
        sc   = r @ self.cents.T / self.T
        soft = torch.softmax(sc, dim=-1)
        hard = soft.argmax(dim=-1)
        return soft, hard

    def load_balance_loss(self, soft):
        """
        §4.3  Lbal = m * sum_p (mean_i pi_ip)^2
        Penalises uneven partition utilisation.
        """
        avg = soft.mean(dim=(0, 1))      # (n_partitions,)
        return self.n * (avg ** 2).sum()


# ─────────────────────────────────────────────────────────────────────────────
#  §4.1  CRPA Sparse Mask Construction
# ─────────────────────────────────────────────────────────────────────────────

def build_crpa_mask(T, p_size, relay_pos, hard_asgn=None,
                    cross_k=4, causal=True, device='cpu'):
    """
    Build CRPA boolean attention mask.
    Omega(i) = P(i) union G union C_k(i)

    Components:
      1. Block-diagonal partition windows
      2. Relay tokens: every token attends to all relays; relays attend globally
      3. Cross-partition routing: top-k random tokens from other partitions
      4. Causal (lower-triangular) constraint

    Fully vectorised — no Python loops over tokens.
    Complexity: O(n(w+g+k))
    """
    mask = torch.zeros(T, T, dtype=torch.bool, device=device)

    # 1. Partition-local attention (block diagonal)
    n_parts = math.ceil(T / p_size)
    for p in range(n_parts):
        s, e = p * p_size, min((p+1) * p_size, T)
        mask[s:e, s:e] = True

    # 2. Relay tokens — Lemma 1 connectivity guarantee
    if relay_pos:
        rp = torch.tensor(relay_pos, device=device)
        mask[:, rp] = True    # all tokens attend to relays
        mask[rp, :] = True    # relays attend to all partitions

    # 3. Cross-partition routing (vectorised topk, no Python loop)
    if hard_asgn is not None and cross_k > 0:
        qi_p = hard_asgn.unsqueeze(1).expand(T, T)
        kj_p = hard_asgn.unsqueeze(0).expand(T, T)
        diff = (qi_p != kj_p)
        causal_lo = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
        scores    = torch.rand(T, T, device=device) * diff.float() * causal_lo.float()
        k         = min(cross_k, T)
        _, topk   = scores.topk(k, dim=1)
        valid     = scores.gather(1, topk) > 0
        rows      = torch.arange(T, device=device).unsqueeze(1).expand_as(topk)
        mask[rows[valid], topk[valid]] = True

    # 4. Causal constraint
    if causal:
        mask &= torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

    mask.fill_diagonal_(True)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
#  §4.4  CRPA Attention Layer
# ─────────────────────────────────────────────────────────────────────────────

class CRPAAttention(nn.Module):
    """
    Full CRPA attention layer implementing §4.1–4.5.

    Key design choices:
      - Top-p support for overlap measurement (adapts to any sparsity level)
      - Within-partition pair sampling (where overlap actually exists)
      - Mask caching (recompute every 20 steps; routing changes slowly)
      - Sensitivity estimation uses LM validation batches
    """

    def __init__(self, d, n_head, block_size, variant, device='cpu'):
        super().__init__()
        assert d % n_head == 0
        self.variant = variant
        self.n_head  = n_head
        self.d_head  = d // n_head
        self.p_size  = CFG['partition_size']
        self.n_rel   = CFG['n_relays']
        self.cross_k = CFG['cross_k']
        self.rho     = CFG['overlap_rho']
        self.device  = device

        self.Wq   = nn.Linear(d, d, bias=False)
        self.Wk   = nn.Linear(d, d, bias=False)
        self.Wv   = nn.Linear(d, d, bias=False)
        self.Wo   = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(CFG['dropout'])

        if 'crpa' in variant:
            n_parts     = math.ceil(block_size / self.p_size)
            self.router = DifferentiableRouter(d, n_parts, CFG['route_temp'])

        self._sens            = {}   # sensitivity cache: (i,j) -> delta
        self._redundant_pairs = []   # pairs confirmed low-sensitivity (safe to penalize)
        self._Alast           = None # last attention matrix for overlap computation
        self._step            = 0
        self._mcache          = None # mask cache

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _relay_pos(self, T):
        step = max(T // (self.n_rel + 1), 1)
        return [step * (i+1) for i in range(self.n_rel) if step*(i+1) < T]

    def _split(self, x):
        B, T, D = x.shape
        return x.reshape(B, T, self.n_head, self.d_head).transpose(1, 2)

    def _merge(self, x):
        B, H, T, d = x.shape
        return x.transpose(1, 2).reshape(B, T, H*d)

    # ── §3.2  Top-p overlap (adapts to sparse attention) ─────────────────────
    def _top_p_support(self, a_row):
        """Return token indices covering rho fraction of attention mass."""
        sorted_a, idx = a_row.sort(descending=True)
        cum    = sorted_a.cumsum(0)
        cutoff = int((cum < self.rho).sum().item()) + 1
        return set(idx[:cutoff].tolist())

    def _jaccard(self, A_mean, i, j):
        """
        Rτ(i,j) using top-p support.
        §3.2: thresholded Jaccard similarity.
        """
        si = self._top_p_support(A_mean[i])
        sj = self._top_p_support(A_mean[j])
        return len(si & sj) / max(len(si | sj), 1)

    # ── §3.3  Causal sensitivity estimation ──────────────────────────────────
    def get_high_overlap_pairs(self, n=16):
        """
        Sample within-partition pairs with high Jaccard overlap.
        Within-partition sampling is essential: cross-partition pairs share
        only relay tokens (low Jaccard by design).
        """
        if self._Alast is None:
            return []
        A = self._Alast.mean(dim=(0, 1))   # (T, T)
        T = A.shape[0]
        pairs = []
        with torch.no_grad():
            for p in range(math.ceil(T / self.p_size)):
                s = p * self.p_size
                e = min(s + self.p_size, T)
                if e - s < 2:
                    continue
                for _ in range(n * 2):
                    i = random.randint(s, e-1)
                    j = random.randint(s, e-1)
                    if i == j:
                        continue
                    ov = self._jaccard(A, i, j)
                    if ov > 0.2:
                        pairs.append((i, j, ov))
            pairs.sort(key=lambda x: -x[2])
        return pairs[:n]

    def update_sensitivity(self, model, nvx, nvy):
        """
        §3.3  Delta_ij = L_needle(M \ E_ij) - L_needle(M)
        Uses needle retrieval loss (last-token CE) so sensitivity reflects
        which overlaps are retrieval-critical, not just LM-useful.
        Pairs with Delta_ij <= eps are stored as _redundant_pairs for penalization.
        """
        pairs = self.get_high_overlap_pairs(n=CFG['sens_n_pairs'])
        if not pairs:
            return

        eps      = CFG['sens_eps']
        redundant = []
        model.eval()
        with torch.no_grad():
            logits0, _ = model(nvx)
            L0 = F.cross_entropy(logits0[:, -1, :], nvy).item()
            for (i, j, _) in pairs:
                model._mask_pair = (i, j)
                logitsm, _       = model(nvx)
                delta            = F.cross_entropy(logitsm[:, -1, :], nvy).item() - L0
                self._sens[(i, j)] = delta
                if delta <= eps:
                    redundant.append((i, j))
                model._mask_pair = None
        self._redundant_pairs = redundant
        model.train()

    # ── §4.4  Redundancy loss ─────────────────────────────────────────────────
    def redundancy_loss(self, A_detached, attn_live, variant):
        """
        Lred = mean over candidate redundant pairs of dot-product overlap.

        crpa_naive:  penalise all high-overlap pairs regardless of sensitivity
        crpa_causal: penalise only pre-identified low-sensitivity pairs from
                     _redundant_pairs (populated by update_sensitivity using
                     needle retrieval loss — so retrieval-critical overlaps are
                     never penalized)
        """
        T = A_detached.shape[0]

        if variant == 'crpa_naive':
            # Penalise all high-overlap pairs (ablation — ignores causal contribution)
            pairs = []
            for p in range(math.ceil(T / self.p_size)):
                s = p * self.p_size
                e = min(s + self.p_size, T)
                if e - s < 2:
                    continue
                for _ in range(8):
                    i = random.randint(s, e-1)
                    j = random.randint(s, e-1)
                    if i != j and self._jaccard(A_detached, i, j) >= 0.2:
                        pairs.append((i, j))
        else:
            # crpa_causal: use pre-identified redundant pairs only
            # These were confirmed low-sensitivity on the needle task —
            # removing them does NOT hurt retrieval, so it is safe to penalize.
            pairs = [(i, j) for i, j in self._redundant_pairs if i < T and j < T]

        if not pairs:
            return torch.tensor(0.0, device=self.device)

        A_live = attn_live.mean(dim=(0, 1))   # (T, T)
        pen = torch.stack([
            (A_live[i] * A_live[j]).sum()
            for i, j in pairs
        ]).mean()
        return pen

    # ── Forward pass ─────────────────────────────────────────────────────────
    def forward(self, x, mask_pair=None):
        """
        Args:
            x:         (B, T, D) token representations
            mask_pair: (i, j) optional — zeros edge i→j for sensitivity est.
        Returns:
            out:  (B, T, D)
            Lb:   load balance loss scalar
            Lr:   redundancy loss scalar
        """
        B, T, D = x.shape
        Q = self._split(self.Wq(x))
        K = self._split(self.Wk(x))
        V = self._split(self.Wv(x))
        scores = (Q @ K.transpose(-2, -1)) * (self.d_head ** -0.5)

        # Build sparse attention mask
        if self.variant == 'dense':
            M = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))

        elif self.variant == 'sliding':
            M = torch.zeros(T, T, dtype=torch.bool, device=x.device)
            for i in range(T):
                M[i, max(0, i - self.p_size + 1):i+1] = True

        else:  # crpa_*
            self._step += 1
            rp         = self._relay_pos(T)
            soft, hard = self.router(x)

            # Mask cache: rebuild every 20 steps (routing changes slowly)
            if (self._mcache is None
                    or self._mcache.shape[0] != T
                    or self._step % 20 == 0):
                self._mcache = build_crpa_mask(
                    T, self.p_size, rp, hard[0].detach(),
                    self.cross_k, causal=True, device=x.device)
            M = self._mcache

        # Apply pair mask for sensitivity estimation
        if mask_pair is not None:
            im, jm = mask_pair
            if 0 <= im < T and 0 <= jm < T:
                M = M.clone()
                M[im, jm] = False

        # Masked softmax
        scores = scores.masked_fill(~M.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn   = torch.softmax(scores, dim=-1)
        attn   = torch.nan_to_num(attn, nan=0.0)
        attn   = self.drop(attn)
        out    = self.Wo(self._merge(attn @ V))

        # Store for overlap analysis
        self._Alast = attn.detach()

        # Auxiliary losses
        Lb = Lr = torch.tensor(0.0, device=x.device)
        if 'crpa' in self.variant:
            soft, _ = self.router(x)
            Lb = self.router.load_balance_loss(soft)
            if self.variant in ('crpa_naive', 'crpa_causal'):
                Am = attn.mean(dim=(0, 1)).detach()
                Lr = self.redundancy_loss(Am, attn, self.variant)

        return out, Lb, Lr


# ─────────────────────────────────────────────────────────────────────────────
#  Full GPT Model
# ─────────────────────────────────────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 4*d), nn.GELU(),
            nn.Linear(4*d, d), nn.Dropout(CFG['dropout']))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, d, n_head, block_size, variant, device='cpu'):
        super().__init__()
        self.attn = CRPAAttention(d, n_head, block_size, variant, device)
        self.ffn  = FFN(d)
        self.ln1  = nn.LayerNorm(d)
        self.ln2  = nn.LayerNorm(d)

    def forward(self, x, mask_pair=None):
        o, lb, lr = self.attn(self.ln1(x), mask_pair)
        x = x + o
        x = x + self.ffn(self.ln2(x))
        return x, lb, lr


class GPT(nn.Module):
    """
    Decoder-only transformer with switchable attention variant.
    Identical backbone across all ablation conditions.
    """
    def __init__(self, variant, block_size, vocab_size, device='cpu'):
        super().__init__()
        d, nh, nl       = CFG['n_embd'], CFG['n_head'], CFG['n_layer']
        self.block_size = block_size
        self.variant    = variant
        self._mask_pair = None    # set during sensitivity estimation

        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(block_size, d)
        self.drop    = nn.Dropout(CFG['dropout'])
        self.blocks  = nn.ModuleList(
            [Block(d, nh, block_size, variant, device) for _ in range(nl)])
        self.ln_f    = nn.LayerNorm(d)
        self.head    = nn.Linear(d, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight   # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x    = self.drop(
            self.tok_emb(idx) +
            self.pos_emb(torch.arange(T, device=idx.device)))

        Lb = Lr = torch.tensor(0.0, device=idx.device)
        for blk in self.blocks:
            x, lb, lr = blk(x, self._mask_pair)
            Lb += lb
            Lr += lr

        x      = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            B2, T2, C = logits.shape
            tl   = F.cross_entropy(
                logits.view(B2 * T2, C), targets.view(B2 * T2))
            loss = (tl
                    + CFG['lambda_bal'] * Lb
                    + CFG['lambda_red'] * Lr)
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
