"""
Checks of the ALS assumptions listed in Section 3 of the report.

Used by check_simulation.py (simulated data, where every assumption holds by
construction, so its output shows what "passing" looks like) and by
run_real.py (MovieLens).

  Assumption 1  low-rank structure     held-out RMSE against k has an interior minimum
  Assumption 3  bilinear interaction   residuals r − r̂ show no trend against r̂
  Assumption 4  λ > 0                  rows with fewer than k ratings; the smallest
                                       eigenvalue of every G is at least λ (Prop. 2)
  Assumption 5  equal-weight errors    shape of the residual distribution, and the
                                       share of Σ e² carried by the largest 5% of errors

numpy.linalg is used here only as a check, never inside the method.

Owner: Muhammad Saad Malik (ALS)
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from methods import my_normal_equations

# For normal errors Z ~ N(0, σ²), the largest 5% of |Z| carry a fixed share of
# E[Z²]:  2 [z φ(z) + (1 − Φ(z))]  at z = Φ⁻¹(0.975) ≈ 1.96,  which is ≈ 0.279.
_z = stats.norm.ppf(0.975)
NORMAL_TOP5_SHARE = 2 * (_z * stats.norm.pdf(_z) + stats.norm.sf(_z))


def interior_minimum(ks, rmses):
    """Assumption 1. The k with the lowest held-out RMSE, and whether it lies
    strictly inside the range of k tried (an interior minimum)."""
    j = int(np.argmin(rmses))
    return ks[j], 0 < j < len(ks) - 1


def rows_below_k(mask, k):
    """Assumption 4. Users and movies with fewer than k known ratings. For these
    rows V_uᵀV_u (or U_iᵀU_i) has rank below k, so it is singular without λ."""
    return int((mask.sum(axis=1) < k).sum()), int((mask.sum(axis=0) < k).sum())


def min_eig_G(R, mask, U, V, lam):
    """Assumption 4 / Proposition 2. Smallest eigenvalue over every user system
    G_u = V_uᵀV_u + λI and every movie system U_iᵀU_i + λI. Prop. 2 says ≥ λ."""
    lowest = np.inf
    for F, rows, M in ((V, R, mask), (U, R.T, mask.T)):
        for r in range(M.shape[0]):
            G, _ = my_normal_equations(F[M[r]], rows[r, M[r]], lam)
            lowest = min(lowest, np.linalg.eigvalsh(G)[0])
    return lowest


def residual_trend(pred, resid, n_bins=10):
    """Assumption 3. Sort the predictions into n_bins groups of equal size and
    take the mean residual in each, plus the calibration slope b of the
    least-squares line r = a + b r̂:
        b = Σ (r̂ − mean r̂)(r − mean r) / Σ (r̂ − mean r̂)².
    No trend means every bin mean ≈ 0 and b ≈ 1; b > 1 means the predictions
    are too timid (pulled too far towards µ), b < 1 that they are too extreme."""
    chunks = np.array_split(np.argsort(pred), n_bins)
    centres = np.array([pred[c].mean() for c in chunks])
    means = np.array([resid[c].mean() for c in chunks])
    actual = pred + resid
    p = pred - pred.mean()
    slope = np.sum(p * (actual - actual.mean())) / np.sum(p ** 2)
    return centres, means, slope


def residual_shape(resid):
    """Assumption 5. Mean, sd, skewness, excess kurtosis (both 0 for a normal
    distribution), and the share of Σ e² carried by the largest 5% of |e|
    (NORMAL_TOP5_SHARE ≈ 0.279 for normal errors)."""
    e = resid - resid.mean()
    sq = np.sort(resid ** 2)[::-1]
    return dict(mean=float(resid.mean()),
                sd=float(resid.std(ddof=1)),
                skew=float(np.mean(e ** 3) / np.mean(e ** 2) ** 1.5),
                kurt=float(np.mean(e ** 4) / np.mean(e ** 2) ** 2 - 3),
                top5=float(sq[: int(np.ceil(0.05 * len(sq)))].sum() / sq.sum()))


def print_checks(curves, mask, k, lam, lowest_eig, pred, resid, true_rank=None):
    """Print one block with all four checks, in the same format for both data sets."""
    print("   Assumption 1, low rank (held-out RMSE against k):")
    for label, (ks, rmses) in curves.items():
        k_low, interior = interior_minimum(ks, rmses)
        where = "interior minimum" if interior else "at the edge of the k range"
        print(f"      {label:<10} " + "  ".join(f"k={kk}: {r:.3f}" for kk, r in zip(ks, rmses))
              + f"   → lowest at k = {k_low} ({where})")
    if true_rank is not None:
        print(f"      (true rank {true_rank})")

    n_u, n_i = rows_below_k(mask, k)
    print(f"   Assumption 4, λ > 0 (k = {k}, λ = {lam:g}):")
    print(f"      users with fewer than k ratings: {n_u};  movies: {n_i}")
    print(f"      smallest eigenvalue of any G: {lowest_eig:.4f}  (Proposition 2: ≥ λ = {lam:g})")

    _, means, slope = residual_trend(pred, resid)
    s = residual_shape(resid)
    print("   Assumption 3, bilinear (residual r − r̂ against r̂, 10 equal-size bins):")
    print("      mean residual per bin: " + ", ".join(f"{x:+.3f}" for x in means))
    print(f"      largest |bin mean| {np.abs(means).max():.3f} (residual sd {s['sd']:.3f});"
          f"  calibration slope {slope:.3f}  (1 = no trend)")

    print("   Assumption 5, equal-weight squared error (residual distribution):")
    print(f"      mean {s['mean']:+.3f}, sd {s['sd']:.3f}, skewness {s['skew']:+.2f}, "
          f"excess kurtosis {s['kurt']:+.2f}  (normal: 0, 0)")
    print(f"      largest 5% of errors carry {s['top5']:.1%} of Σ e²  "
          f"(normal errors: {NORMAL_TOP5_SHARE:.1%})")


def print_residual_table(rows):
    """Assumptions 3 and 5 side by side for several predictors, given as
    (label, predictions, residuals). Used to separate what the data does from
    what the estimator adds."""
    print(f"      {'predictor':<36}{'slope':>7}{'max|bin|':>10}{'kurtosis':>10}{'top 5%':>8}")
    for label, pred, resid in rows:
        _, means, slope = residual_trend(pred, resid)
        s = residual_shape(resid)
        print(f"      {label:<36}{slope:>7.3f}{np.abs(means).max():>10.3f}"
              f"{s['kurt']:>+10.2f}{s['top5']:>8.1%}")


def plot_checks(curves, pred, resid, title, path):
    """One figure, three panels: Assumption 1 (RMSE against k), Assumption 3
    (residual against prediction) and Assumption 5 (residual histogram)."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    ax = axes[0]
    for label, (ks, rmses) in curves.items():
        ax.plot(ks, rmses, marker="o", label=label)
    ax.set_xticks(sorted({kk for ks, _ in curves.values() for kk in ks}))
    ax.set_xlabel("Number of factors k")
    ax.set_ylabel("Held-out RMSE")
    ax.set_title("Assumption 1: RMSE against k")
    if len(curves) > 1:
        ax.legend(fontsize=8)

    ax = axes[1]
    ax.hexbin(pred, resid, gridsize=45, cmap="Blues", mincnt=1, bins="log")
    centres, means, _ = residual_trend(pred, resid)
    ax.plot(centres, means, color="darkorange", marker="o", label="mean per bin")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction r̂")
    ax.set_ylabel("Residual r − r̂")
    ax.set_title("Assumption 3: residual against prediction")
    ax.legend(fontsize=8)

    ax = axes[2]
    s = residual_shape(resid)
    ax.hist(resid, bins=60, density=True, color="steelblue", alpha=0.8, label="residuals")
    x = np.linspace(resid.min(), resid.max(), 300)
    ax.plot(x, stats.norm.pdf(x, s["mean"], s["sd"]), color="darkorange", label="normal, same mean and sd")
    ax.set_xlabel("Residual r − r̂")
    ax.set_ylabel("Density")
    ax.set_title("Assumption 5: residual distribution")
    ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
