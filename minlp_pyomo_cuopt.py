"""
Solving the simple MINLP from minlp_pyomo.py with cuOpt (GPU Linear MIP).

cuOpt supports LP and Mixed-Integer Linear Programming only — no non-linear
constraints or objectives.  Both non-linear components must be reformulated:

  Component          Original (Bonmin)          cuOpt reformulation
  ─────────────────────────────────────────────────────────────────────
  Objective          (x−1)² + 2(y−1)² + (z−1.5)²   L1: |x−1| + 2|y−1| + |z−1.5|
                     quadratic                       linear via aux vars dx, dy, dz
  Ball constraint    x² + y² ≤ 3 − 0.5z             outer-approximation tangent planes
                     non-linear convex               linear (one plane per support point)

Outer approximation (OA) of x² + y² + 0.5z ≤ 3
──────────────────────────────────────────────────
The constraint is convex in (x, y, z).  Any tangent hyperplane at (x₀, y₀)
is a valid linear outer approximation:

    2x₀·x + 2y₀·y + 0.5z  ≤  3 + x₀² + y₀²

The union of all such half-spaces is exactly the original convex set;
a finite selection gives an inner approximation of the feasible region.

N+1 support points are chosen at radius R=1.5 evenly spaced in [0, π/2]:
  θₖ = k·π/(2N),  x₀ = R·cos(θ),  y₀ = R·sin(θ)
  → RHS = 3 + R² = 5.25  (constant for all cuts)

Known global optimum (both formulations reach the same solution point):
  x=1, y=1, z=1 or z=2  (tie: |1−1.5| = |2−1.5| = 0.5)
  L2 objective (Bonmin):  0 + 0 + 0.25 = 0.25
  L1 objective (cuOpt):   0 + 0 + 0.50 = 0.50
"""

import math
import time

from cuopt.linear_programming.problem import Problem, VType
from cuopt import linear_programming as lp

# ── OA cut parameters ────────────────────────────────────────────────────────
_OA_RADIUS    = 1.5  # tangent support circle radius
_OA_N_CUTS    = 8    # cuts at angles 0, π/(2N), 2π/(2N), … π/2  (N+1 total)


def build_model() -> tuple:
    p = Problem("Simple_MINLP_linearised")

    # ── Primary decision variables (same bounds as minlp_pyomo.py) ──────────
    x = p.addVariable(lb=0.0, ub=3.0, vtype=VType.CONTINUOUS, name="x")
    y = p.addVariable(lb=0.0, ub=3.0, vtype=VType.CONTINUOUS, name="y")
    z = p.addVariable(lb=0.0, ub=4.0, vtype=VType.INTEGER,    name="z")

    # ── Auxiliary variables for L1 linearisation of the objective ───────────
    dx = p.addVariable(lb=0.0, ub=2.0, vtype=VType.CONTINUOUS, name="dx")  # |x-1|
    dy = p.addVariable(lb=0.0, ub=2.0, vtype=VType.CONTINUOUS, name="dy")  # |y-1|
    dz = p.addVariable(lb=0.0, ub=3.0, vtype=VType.CONTINUOUS, name="dz")  # |z-1.5|

    # ── Objective: min dx + 2·dy + dz  (weights match original) ─────────────
    p.setObjective(dx + 2 * dy + dz)

    # |x − 1|
    p.addConstraint(dx >= x - 1, name="dx_upper")
    p.addConstraint(dx >= 1 - x, name="dx_lower")

    # |y − 1|
    p.addConstraint(dy >= y - 1, name="dy_upper")
    p.addConstraint(dy >= 1 - y, name="dy_lower")

    # |z − 1.5|
    p.addConstraint(dz >= z - 1.5, name="dz_upper")
    p.addConstraint(dz >= 1.5 - z, name="dz_lower")

    # ── Outer-approximation of  x² + y² + 0.5z ≤ 3 ─────────────────────────
    # Tangent at (x₀, y₀): 2x₀·x + 2y₀·y + 0.5z ≤ 3 + x₀² + y₀²
    R   = _OA_RADIUS
    rhs = 3.0 + R ** 2   # 3 + 2.25 = 5.25 (same for all support points on this circle)
    for k in range(_OA_N_CUTS + 1):
        theta = math.pi / 2 * k / _OA_N_CUTS
        x0    = R * math.cos(theta)
        y0    = R * math.sin(theta)
        p.addConstraint(
            2 * x0 * x + 2 * y0 * y + 0.5 * z <= rhs,
            name=f"ball_{k}",
        )

    # ── Linear coupling (identical to minlp_pyomo.py) ────────────────────────
    p.addConstraint(x + y + z >= 2, name="coupling")

    return p, x, y, z


# ── Solve ────────────────────────────────────────────────────────────────────

def solve(model, tee=False):
    settings = lp.SolverSettings()
    settings.set_parameter("time_limit", 60.0)
    settings.set_parameter("log_to_console", 1 if tee else 0)

    t0 = time.perf_counter()
    model.solve(settings)
    return time.perf_counter() - t0


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("MINLP → Linear MIP  (cuOpt 26.4.0 · GPU)")
    print("=" * 55)
    print("  Original (Bonmin/MINLP):")
    print("    min  (x-1)² + 2(y-1)² + (z-1.5)²")
    print("    s.t. x² + y²  ≤  3 - 0.5z")
    print("         x + y + z ≥  2")
    print("         0 ≤ x,y ≤ 3  (continuous);  z ∈ {0..4}  (integer)")
    print()
    print("  cuOpt reformulation (Linear MIP):")
    print("    min  dx + 2·dy + dz               [L1 objective]")
    print("    s.t. 2x₀x + 2y₀y + 0.5z ≤ 5.25   [OA tangent planes × 9]")
    print("         x + y + z ≥ 2                 [original coupling]")
    print(f"  OA: {_OA_N_CUTS + 1} tangent planes at radius {_OA_RADIUS} in [0, π/2]")
    print()

    t_build0 = time.perf_counter()
    model, x_var, y_var, z_var = build_model()
    t_build = time.perf_counter() - t_build0

    elapsed = solve(model)

    status_map = {1: "optimal", 0: "infeasible", 2: "unbounded", 3: "timelimit"}
    status = status_map.get(model.Status, f"unknown({model.Status})")

    xv = x_var.Value
    yv = y_var.Value
    zv = round(z_var.Value)   # round integer to exact int

    # Recompute both objective forms at the solution
    l1_obj = abs(xv - 1) + 2 * abs(yv - 1) + abs(zv - 1.5)
    l2_obj = (xv - 1) ** 2 + 2 * (yv - 1) ** 2 + (zv - 1.5) ** 2

    ball_slack    = (3 - 0.5 * zv) - (xv ** 2 + yv ** 2)
    coupling_slack = xv + yv + zv - 2

    print(f"Solver status             : {status}")
    print(f"MIP gap                   : {model.SolutionStats.mip_gap:.2e}")
    print()
    print(f"  x  = {xv:.6f}")
    print(f"  y  = {yv:.6f}")
    print(f"  z  = {zv}")
    print()
    print(f"  Objective  (L1, cuOpt)  = {model.ObjValue:.6f}")
    print(f"  Objective  (L2, Bonmin) = {l2_obj:.6f}   ← same solution, different metric")
    print()
    print(f"  Ball slack  (3 − 0.5z − x² − y²) : {ball_slack:+.6f}")
    print(f"  Coupling slack (x+y+z − 2)        : {coupling_slack:+.6f}")
    print()
    print(f"Wall-clock  build = {t_build*1e3:.1f} ms   "
          f"solve = {elapsed*1e3:.1f} ms   "
          f"total = {(t_build+elapsed)*1e3:.1f} ms")
