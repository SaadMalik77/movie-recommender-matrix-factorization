"""
Simulated rating data with known truth.

Factor model (used to test ALS and the truncated SVD):
  R_true = A B,  A ∈ R^{m×k} and B ∈ R^{k×n} with standard normal entries (true rank k);
  r_ui = (R_true)_ui + ε_ui,  ε_ui ~ N(0, noise²);
  each entry is observed independently with probability frac (the set Ω).

With the defaults (m = 100, n = 80, k = 3, frac = 0.3, noise = 0.1, seed = 0)
this draws exactly the data of the report's Simulation Check: |Ω| = 2,281.
"""
import numpy as np


def simulate_ratings(m=100, n=80, k=3, frac=0.3, noise=0.1, seed=0):
    """Return the noisy ratings R, the true ratings R_true and the mask of Ω."""
    rng = np.random.default_rng(seed)
    R_true = rng.normal(size=(m, k)) @ rng.normal(size=(k, n))
    R = R_true + noise * rng.normal(size=(m, n))
    mask = rng.random((m, n)) < frac
    return R, R_true, mask
