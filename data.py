"""
data.py — WikiText-2 dataset and Needle-in-Haystack retrieval task.

WikiText-2 is used for language modelling evaluation (comparable PPL
numbers with prior work: Longformer, BigBird, etc.).

Needle-in-Haystack synthetic task:
  - Embed n key→value pairs at random positions in filler tokens.
  - Query the model with a key at the last position.
  - Target: model predicts the correct value token.
  - Controlled test of long-range retrieval.
"""

import random
import torch
from config import CFG

# ── These are populated by init_data() ──────────────────────────────────────
tokenizer   = None
vocab_size  = None
train_data  = None
val_data    = None


def init_data(device='cpu'):
    """
    Download and tokenize WikiText-2.
    Must be called once before get_lm_batch / make_needle_batch.
    """
    global tokenizer, vocab_size, train_data, val_data

    try:
        from datasets import load_dataset
        from transformers import GPT2TokenizerFast
    except ImportError:
        raise ImportError(
            "Run: pip install datasets transformers"
        )

    print("Loading GPT2 tokenizer...")
    tokenizer  = GPT2TokenizerFast.from_pretrained('gpt2')
    vocab_size = tokenizer.vocab_size   # 50257

    print("Loading WikiText-2...")
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1')

    def tok(split):
        raw = '\n'.join(ds[split]['text'])
        ids = tokenizer.encode(raw)
        return torch.tensor(ids, dtype=torch.long)

    train_data = tok('train')
    val_data   = tok('validation')

    # Sanity checks
    assert CFG['val_range'][1] < vocab_size, \
        "Needle token ranges exceed vocab size"

    print(f"Vocab: {vocab_size} | "
          f"Train: {len(train_data):,} | "
          f"Val: {len(val_data):,}")
    return tokenizer, vocab_size, train_data, val_data


def get_lm_batch(split, block_size, bs=None, device='cpu'):
    """Sample a batch for language modelling."""
    bs  = bs or CFG['batch_size']
    src = train_data if split == 'train' else val_data
    ix  = torch.randint(len(src) - block_size, (bs,))
    x   = torch.stack([src[i : i+block_size]     for i in ix])
    y   = torch.stack([src[i+1 : i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


def make_needle_batch(block_size, bs=None, n_needles=2,
                      needle_depth=None, device='cpu'):
    """
    Needle-in-Haystack batch.

    Each sequence embeds n key→value pairs in filler tokens.
    The last token is a query key; the target is its paired value.

    Args:
        block_size:    sequence length
        bs:            batch size
        n_needles:     number of key-value pairs to embed
        needle_depth:  float 0-1, force needle at this sequence fraction
                       (used for depth-sweep evaluation)
        device:        torch device
    """
    bs     = bs or CFG['batch_size']
    fr, kr, vr = CFG['filler_range'], CFG['key_range'], CFG['val_range']
    n_keys = min(n_needles, kr[1] - kr[0] + 1)
    seqs, targets = [], []

    for _ in range(bs):
        keys = random.sample(range(kr[0], kr[1]+1), n_keys)
        vals = [random.randint(*vr) for _ in keys]
        kv   = dict(zip(keys, vals))
        seq, inserted = [], set()
        kv_items = list(kv.items())

        # One partition away: needle in partition 2 (tokens 256-383), query in partition 3 (511)
        # Single relay hop: relay@306 picks up needle → relay@408 → query@511
        effective_depth = needle_depth if needle_depth is not None else random.uniform(0.55, 0.73)

        while len(seq) < block_size - 3:
            remaining = [p for p in kv_items if p[0] not in inserted]
            tgt       = int(effective_depth * block_size)
            insert_ok = (remaining
                         and len(seq) >= tgt
                         and len(seq) < tgt + 10)

            if insert_ok:
                k, v = random.choice(remaining)
                seq.extend([k, v])
                inserted.add(k)
            else:
                seq.append(random.randint(*fr))

        seen = list(inserted)
        if not seen:
            seq.append(random.randint(*kr))
            targets.append(random.randint(*vr))
        else:
            qk = random.choice(seen)
            seq.append(qk)
            targets.append(kv[qk])

        seq = seq[:block_size]
        while len(seq) < block_size:
            seq.append(random.randint(*fr))
        seqs.append(seq)

    x = torch.tensor(seqs,    dtype=torch.long, device=device)
    y = torch.tensor(targets, dtype=torch.long, device=device)
    return x, y
