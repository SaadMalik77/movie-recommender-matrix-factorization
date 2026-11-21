"""
Alternating least squares (ALS) for matrix factorization, written from scratch.

Model:      R ≈ U Vᵀ,  U ∈ R^{n_u × k} (row u_u = user u's factors),
                       V ∈ R^{n_i × k} (row v_i = movie i's factors)
Objective:  J(U, V) = Σ_{(u,i)∈Ω} (r_ui − u_uᵀ v_i)² + λ (‖U‖_F² + ‖V‖_F²)

The core method uses only NumPy arrays, matrix products and our own Gaussian
elimination. numpy.linalg appears only in the verification scripts.

Every function also works on NumPy arrays of fractions.Fraction (dtype=object),
so the small worked example is computed exactly by the same code that runs
the simulation and the real data.

Owner: Muhammad Saad Malik (ALS)
"""
import numpy as np


# Step 0. Initialize the factors.
#   U⁽⁰⁾ ∈ R^{n_u × k},  V⁽⁰⁾ ∈ R^{n_i × k},  entries drawn from N(0, 1).
def my_init_factors(n_users, n_items, k, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n_users, k))
    V = rng.normal(size=(n_items, k))
    return U, V


# Normal equations for one row (one user or one movie):
#   A = Fᵀ F + λ I   (k × k),    b = Fᵀ r   (length k)
# For user u:  F = V_u (rows of V for the movies u rated),   r = r_u (u's ratings).
# For movie i: F = U_i (rows of U for the users who rated i), r = r_i (ratings of i).
def my_normal_system(F, r, lam):
    k = F.shape[1]
    A = F.T @ F + lam * np.eye(k, dtype=F.dtype)
    b = F.T @ r
    return A, b


# Solve A x = b by Gaussian elimination with partial pivoting.
#   Forward:  for each column c, swap up the row with the largest |A[r, c]| (r ≥ c),
#             then subtract (A[r, c] / A[c, c]) × row c from every row r > c.
#   Backward: x_r = (b_r − Σ_{j>r} A[r, j] x_j) / A[r, r],   r = k−1, …, 0.
def my_gauss_solve(A, b):
    n = A.shape[0]
    M = np.empty((n, n + 1), dtype=np.result_type(A, b, np.float64))
    M[:, :n] = A
    M[:, n] = b
    for c in range(n):
        p = c + int(np.argmax([abs(M[r, c]) for r in range(c, n)]))
        if M[p, c] == 0:
            raise ValueError("matrix is singular")
        if p != c:
            M[[c, p]] = M[[p, c]]
        for r in range(c + 1, n):
            M[r, c:] = M[r, c:] - (M[r, c] / M[c, c]) * M[c, c:]
    x = np.empty(n, dtype=M.dtype)
    for r in range(n - 1, -1, -1):
        s = M[r, n]
        for j in range(r + 1, n):
            s = s - M[r, j] * x[j]
        x[r] = s / M[r, r]
    return x


# Step 1. User update with V fixed. For every user u:
#   u_u = (V_uᵀ V_u + λ I)⁻¹ V_uᵀ r_u
# computed by solving the k × k system, never by forming the inverse.
def my_update_U(R, mask, V, lam):
    n_users, k = R.shape[0], V.shape[1]
    U = np.empty((n_users, k), dtype=np.result_type(R, V, np.float64))
    for u in range(n_users):
        seen = mask[u]                       # movies rated by user u
        A, b = my_normal_system(V[seen], R[u, seen], lam)
        U[u] = my_gauss_solve(A, b)
    return U


# Step 2. Movie update with U fixed. For every movie i:
#   v_i = (U_iᵀ U_i + λ I)⁻¹ U_iᵀ r_i
# This is Step 1 applied to the transposed problem Rᵀ ≈ V Uᵀ.
def my_update_V(R, mask, U, lam):
    return my_update_U(R.T, mask.T, U, lam)


# Step 3. Objective, split into its two parts:
#   J = Σ_{(u,i)∈Ω} (r_ui − u_uᵀ v_i)²   +   λ (‖U‖_F² + ‖V‖_F²)
#       └──────── squared error ───────┘     └──── penalty ────┘
def my_objective(R, mask, U, V, lam, parts=False):
    E = R - U @ V.T
    sse = np.sum(E[mask] ** 2)
    penalty = lam * (np.sum(U ** 2) + np.sum(V ** 2))
    return (sse + penalty, sse, penalty) if parts else sse + penalty


# Full ALS: Step 0, then repeat Steps 1–3.
#   J_hist[0] is J at the random start; J_hist[t] is J after iteration t.
def my_als(R, mask, k, lam, iters=50, seed=0):
    U, V = my_init_factors(R.shape[0], R.shape[1], k, seed)
    J_hist = [my_objective(R, mask, U, V, lam)]
    for _ in range(iters):
        U = my_update_U(R, mask, V, lam)
        V = my_update_V(R, mask, U, lam)
        J_hist.append(my_objective(R, mask, U, V, lam))
    return U, V, np.array(J_hist)


# Prediction for every cell:  R̂ = U Vᵀ,  r̂_ui = u_uᵀ v_i
def my_predict(U, V):
    return U @ V.T


# Comparison helper for every verification:
#   max |mine − ref|   and   relative error ‖mine − ref‖ / ‖ref‖
def report(name, mine, ref):
    mine, ref = np.asarray(mine, float), np.asarray(ref, float)
    abs_err = np.abs(mine - ref).max()
    rel_err = np.linalg.norm(mine - ref) / np.linalg.norm(ref)
    print(f"{name:32s} max|diff| = {abs_err:.2e}   rel = {rel_err:.2e}")
