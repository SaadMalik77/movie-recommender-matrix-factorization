"""
Simulated rating data with known truth.

Factor model (used to test ALS and the truncated SVD):
  R_true = A B,  A ∈ R^{m×k} and B ∈ R^{k×n} with standard normal entries (true rank k);
  r_ui = (R_true)_ui + ε_ui,  ε_ui ~ N(0, noise²);
  each entry is observed independently with probability frac (the set Ω).

With the defaults (m = 100, n = 80, k = 3, frac = 0.3, noise = 0.1, seed = 0)
this draws exactly the data of the report's Simulation Check: |Ω| = 2,281.

Bias model (used to test Method 1):
  r_ui = µ + b_u + c_i + ε_ui on a given Ω, with b_u ~ N(0, 0.3²), c_i ~ N(0, 0.5²),
  ε_ui ~ N(0, 0.3²) and µ = 3.5, as in the report.

Error model (used to test the confidence interval of Proposition 8):
  N prediction errors from N(0, rmse²), so the true MSE is rmse² and the true RMSE is rmse.
"""
import numpy as np


def simulate_ratings(m=100, n=80, k=3, frac=0.3, noise=0.1, seed=0):
    """Return the noisy ratings R, the true ratings R_true and the mask of Ω."""
    rng = np.random.default_rng(seed)
    R_true = rng.normal(size=(m, k)) @ rng.normal(size=(k, n))
    R = R_true + noise * rng.normal(size=(m, n))
    mask = rng.random((m, n)) < frac
    return R, R_true, mask


def simulate_bias_ratings(mask, mu=3.5, user_sd=0.3, movie_sd=0.5, noise=0.3, seed=0):
    """Return the noisy ratings R, the true ratings R_true and the true biases b, c."""
    rng = np.random.default_rng(seed)
    m, n = mask.shape
    b = user_sd * rng.normal(size=m)
    c = movie_sd * rng.normal(size=n)
    R_true = mu + b[:, None] + c[None, :]
    R = R_true + noise * rng.normal(size=(m, n))
    return R, R_true, b, c


def simulate_test_errors(rng, n_test=20_000, rmse=0.9):
    """One experiment of the error model: n_test errors from N(0, rmse²)."""
    return rng.normal(scale=rmse, size=n_test)
