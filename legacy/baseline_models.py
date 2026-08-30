# baseline_models.py

import numpy as np

def random_baseline(grid):
    return np.random.rand(*grid.shape)

def historical_baseline(real_grid):
    return real_grid / real_grid.max()