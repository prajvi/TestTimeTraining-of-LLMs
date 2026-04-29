"""
Bootstrap confidence intervals for scalar metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap CI for the mean of `values`.

    Returns:
      (mean, lower, upper)
    """
    if len(values) == 0:
        raise ValueError("bootstrap_ci: empty values")

    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())

    n = len(arr)
    idxs = rng.integers(0, n, size=(n_resamples, n))
    samples = arr[idxs].mean(axis=1)

    alpha = 1.0 - ci
    lower = float(np.quantile(samples, alpha / 2.0))
    upper = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return mean, lower, upper

