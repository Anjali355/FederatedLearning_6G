import os, random
import numpy as np
import torch
import pandas as pd

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def minmax_scale(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32)
    mins = X.min(axis=0, keepdims=True)
    maxs = X.max(axis=0, keepdims=True)
    denom = np.where((maxs - mins) == 0, 1.0, (maxs - mins))
    return (X - mins) / denom

def make_partitions(X, y, K, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    X, y = X[idx], y[idx]

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0); rng.shuffle(idx1)
    splits0 = np.array_split(idx0, K)
    splits1 = np.array_split(idx1, K)

    parts = []
    for k in range(K):
        part = np.concatenate([splits0[k], splits1[(k + (k%3)) % K]])
        rng.shuffle(part)
        parts.append(part)
    return parts
