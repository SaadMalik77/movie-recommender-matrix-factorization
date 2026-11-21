"""
Small worked example of ALS, computed exactly with fractions.

Data:  a 3 × 3 rating matrix with 2 missing entries, rank k = 1, λ = 1.
Start: v⁽⁰⁾ = (1, 1, 1). Then one full ALS iteration:
   Step 1  update every user  (V fixed):   u_u = (V_uᵀ V_u + λI)⁻¹ V_uᵀ r_u
   Step 2  update every movie (U fixed):   v_i = (U_iᵀ U_i + λI)⁻¹ U_iᵀ r_i
   Step 3  objective J after each step
Every intermediate result is printed. The functions come from methods.py,
the same ones that run the simulation and the real data.

Second-way checks:
   (a) substitute each new row back into its normal equations: A x − b = 0 exactly
       (equivalently, ∂J/∂x = 2(A x − b) = 0, so each update is a stationary point);
   (b) the objective does not increase.

Run from the repository root:   python src/small_example.py
"""
from fractions import Fraction
import numpy as np

from methods import (my_normal_system, my_gauss_solve, my_update_U,
                     my_update_V, my_objective, my_predict)

USERS = ["user 1", "user 2", "user 3"]
MOVIES = ["movie 1", "movie 2", "movie 3"]


def num(x):
    """Exact fraction, with a decimal when it is not a whole number."""
    x = Fraction(x)
    return str(x) if x.denominator == 1 else f"{x} (≈ {float(x):.4f})"


def vec(a):
    """Short exact form of a 1-D or 2-D array, e.g. [5, 4] or [[3]]."""
    a = np.asarray(a, dtype=object)
    if a.ndim == 1:
        return "[" + ", ".join(str(Fraction(x)) for x in a) + "]"
    return "[" + ", ".join(vec(row) for row in a) + "]"


def show_table(title, M, mask=None, mark=None):
    """Print a users × movies table; '?' for missing cells, '*' for predicted ones."""
    print(title)
    cells = [[("?" if mask is not None and not mask[u, i] and mark is None
               else str(Fraction(M[u, i])) + ("*" if mark is not None and not mark[u, i] else ""))
              for i in range(M.shape[1])] for u in range(M.shape[0])]
    w = max(9, max(len(c) for row in cells for c in row) + 2)
    print(" " * 10 + "".join(f"{m:>{w}}" for m in MOVIES))
    for u, row in enumerate(cells):
        print(f"{USERS[u]:10s}" + "".join(f"{c:>{w}}" for c in row))
    print()


def show_J(label, R, mask, U, V, lam):
    J, sse, pen = my_objective(R, mask, U, V, lam, parts=True)
    print(f"{label}")
    print(f"  squared error  Σ_Ω (r_ui − u_u v_i)² = {num(sse)}")
    print(f"  penalty        λ(‖U‖² + ‖V‖²)       = {num(pen)}")
    print(f"  J                                    = {num(J)}\n")
    return J


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
ratings = [[5,    4,    None],
           [4,    None, 2   ],
           [1,    2,    1   ]]

mask = np.array([[r is not None for r in row] for row in ratings])       # Ω
R = np.array([[Fraction(r if r is not None else 0) for r in row]
              for row in ratings], dtype=object)                           # 0 where missing (never used)
k, lam = 1, Fraction(1)
V0 = np.array([[Fraction(1)], [Fraction(1)], [Fraction(1)]], dtype=object)

print("=" * 64)
print("SETUP")
print("=" * 64)
show_table("Rating matrix R (? = missing):", R, mask)
print(f"k = {k},  λ = {lam},  |Ω| = {mask.sum()} known ratings")
print(f"Initial movie factors v⁽⁰⁾ = {vec(V0[:, 0])}\n")

# ---------------------------------------------------------------------------
# Step 1: update every user with V fixed
# ---------------------------------------------------------------------------
print("=" * 64)
print("STEP 1  User update (V fixed):  u_u = (V_uᵀ V_u + λI)⁻¹ V_uᵀ r_u")
print("=" * 64)
for u in range(3):
    seen = mask[u]
    Vu, ru = V0[seen], R[u, seen]
    A, b = my_normal_system(Vu, ru, lam)
    x = my_gauss_solve(A, b)
    rated = ", ".join(m for m, s in zip(MOVIES, seen) if s)
    print(f"{USERS[u]} rated {rated}")
    print(f"  V_u = {vec(Vu)}   (rows of V for those movies)")
    print(f"  r_u = {vec(ru)}   (ratings given)")
    print(f"  A = V_uᵀ V_u + λI = {vec(A)}")
    print(f"  b = V_uᵀ r_u      = {vec(b)}")
    print(f"  solve A u_u = b   →  u_u = {num(x[0])}\n")

U1 = my_update_U(R, mask, V0, lam)
print(f"U⁽¹⁾ = {vec(U1[:, 0])}\n")
J_after_users = show_J("Objective after Step 1  (U⁽¹⁾, V⁽⁰⁾):", R, mask, U1, V0, lam)

# ---------------------------------------------------------------------------
# Step 2: update every movie with U fixed (using the new U⁽¹⁾)
# ---------------------------------------------------------------------------
print("=" * 64)
print("STEP 2  Movie update (U fixed):  v_i = (U_iᵀ U_i + λI)⁻¹ U_iᵀ r_i")
print("=" * 64)
for i in range(3):
    seen = mask[:, i]
    Ui, ri = U1[seen], R[seen, i]
    A, b = my_normal_system(Ui, ri, lam)
    x = my_gauss_solve(A, b)
    raters = ", ".join(u for u, s in zip(USERS, seen) if s)
    print(f"{MOVIES[i]} rated by {raters}")
    print(f"  U_i = {vec(Ui)}   (rows of U for those users)")
    print(f"  r_i = {vec(ri)}   (ratings received)")
    print(f"  A = U_iᵀ U_i + λI = {vec(A)}")
    print(f"  b = U_iᵀ r_i      = {vec(b)}")
    print(f"  solve A v_i = b   →  v_i = {num(x[0])}\n")

V1 = my_update_V(R, mask, U1, lam)
print(f"V⁽¹⁾ = {vec(V1[:, 0])}\n")
J_after_items = show_J("Objective after Step 2  (U⁽¹⁾, V⁽¹⁾):", R, mask, U1, V1, lam)

# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
print("=" * 64)
print("PREDICTIONS  R̂ = U Vᵀ  after one iteration (* = was missing)")
print("=" * 64)
R_hat = my_predict(U1, V1)
show_table("", R_hat, mark=mask)
for u in range(3):
    for i in range(3):
        if not mask[u, i]:
            print(f"  {USERS[u]}, {MOVIES[i]}:  {num(U1[u, 0])} × {num(V1[i, 0])} = {num(R_hat[u, i])}")
print()

# ---------------------------------------------------------------------------
# Second-way checks
# ---------------------------------------------------------------------------
print("=" * 64)
print("CHECKS")
print("=" * 64)
res_users = [my_normal_system(V0[mask[u]], R[u, mask[u]], lam) for u in range(3)]
res_users = [A @ U1[u] - b for u, (A, b) in enumerate(res_users)]
res_items = [my_normal_system(U1[mask[:, i]], R[mask[:, i], i], lam) for i in range(3)]
res_items = [A @ V1[i] - b for i, (A, b) in enumerate(res_items)]
print("(a) Substitute back into the normal equations, A x − b:")
print(f"    users:  {[vec(r) for r in res_users]}")
print(f"    movies: {[vec(r) for r in res_items]}")
ok_a = all(x == 0 for r in res_users + res_items for x in r)
print(f"    all exactly zero: {ok_a}\n")

print("(b) Objective never increases:")
print(f"    J after Step 1 = {num(J_after_users)}")
print(f"    J after Step 2 = {num(J_after_items)}")
ok_b = J_after_items <= J_after_users
print(f"    J did not increase: {ok_b}\n")

# Further iterations in floating point (exact fractions grow very long),
# using the same functions, to show J keeps decreasing.
Rf, U, V = R.astype(float), U1.astype(float), V1.astype(float)
J_hist = [float(J_after_items)]
for _ in range(9):
    U = my_update_U(Rf, mask, V, 1.0)
    V = my_update_V(Rf, mask, U, 1.0)
    J_hist.append(my_objective(Rf, mask, U, V, 1.0))
print("(c) J over iterations 1–10 (floating point):")
print("    " + ", ".join(f"{j:.4f}" for j in J_hist))
print(f"    non-increasing: {bool(np.all(np.diff(J_hist) <= 1e-12))}")
