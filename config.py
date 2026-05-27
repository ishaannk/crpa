"""
config.py — CRPA hyperparameters
Fast ~1hr config: ~8M params, block_size=512, 6 layers.
Paper scale: 138M params, 64k context, A100 80GB.
"""

CFG = dict(
    # Architecture — small scale for fast iteration
    n_embd         = 192,
    n_head         = 8,
    n_layer        = 6,
    dropout        = 0.10,

    # CRPA §4 hyperparameters
    partition_size = 128,
    n_relays       = 4,
    cross_k        = 4,
    route_temp     = 0.70,
    lambda_bal     = 0.01,
    lambda_red     = 0.05,
    overlap_rho    = 0.60,
    sens_eps       = 0.03,
    sens_interval  = 200,
    sens_n_pairs   = 8,

    # Training
    batch_size     = 16,
    max_iters      = 4000,
    eval_iters     = 20,
    eval_interval  = 400,
    lr             = 3e-4,
    weight_decay   = 0.10,
    grad_clip      = 1.0,
    warmup_steps   = 200,
    ret_ratio      = 0.90,
    seed           = 42,

    # Evaluation
    ablation_block_size = 512,
    scale_lens          = [64, 128, 256, 512],
    needle_depths       = [0.1, 0.3, 0.5, 0.7, 0.9],
    multi_seeds         = [42, 1337, 2024],

    # Needle token ranges — 20 keys × 20 values (5% random chance vs 0.2%)
    filler_range   = (2000, 9999),
    key_range      = (100,  119),
    val_range      = (120,  139),
)
