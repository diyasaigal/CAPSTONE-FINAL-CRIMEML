"""Import this FIRST (before torch) in any RL entry point.

The MLP policy is tiny; capping torch to one thread avoids it fighting NumPy for
cores during rollout collection (the env step is the bottleneck and is NumPy).
NumPy/BLAS threading is left alone -- with the in-process DummyVecEnv it helps.
"""
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass
