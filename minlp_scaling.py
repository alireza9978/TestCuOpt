"""
Scalable MINLP benchmark — Pyomo + Bonmin (B-OA algorithm)

Generalised problem for size (n, k):

  min   Σ_{i=1..n}(x_i - 1)²  +  Σ_{j=1..k}(z_j - 1.3)²

  s.t.  x_i²  ≤  1.6 - 0.5·z_{f(i)}   for each i   [separable non-linear]
        Σ x_i  +  Σ z_j  ≥  n + k - 1               [linear coupling]
        x_i ∈ [0, 3]   (continuous)
        z_j ∈ {0,1,2,3}  (integer)

Constraint map:  f(i) = (i-1)*k//n + 1   (evenly distributes x_i among z_j groups)

Why this formulation works at scale
────────────────────────────────────
Separable non-linear constraints:
  • Each cut in B-OA's MILP master touches only 2 variables (x_i, z_{f(i)}).
    → MILP stays sparse regardless of n.
  • IPOPT solves each NLP sub-problem in O(n) time (sparse diagonal Hessian).

RHS = 1.6 (not 1.5):
  • At z_j=1 (integer optimal) → x_i ≤ √1.1 ≈ 1.049.
    The optimal x_i=1 lies in the STRICT INTERIOR (slack 0.1).
  • IPOPT converges fast when the optimum is not on a constraint boundary.
  • At z_j=1.3 (NLP relaxation) → x_i ≤ √0.95 ≈ 0.975.
    B-OA needs only ≈2 outer iterations to converge.

Integer target = 1.3 (not 1.5):
  • Unique nearest integer is 1: (1-1.3)² = 0.09 vs (2-1.3)² = 0.49.
  • No branching symmetry → no combinatorial explosion.

k scaling strategy
──────────────────
k grows slowly (1→4) to keep the MILP master tractable.
n grows ~3× per step to show clear NLP timing progression.
Diagnostic data used to calibrate each (n,k) pair for < 30s solve time.

Known optimum: x_i=1, z_j=1 for all i,j,  f* = 0.09·k
"""

import os
import time

import pyomo.environ as pyo

# ── Library path ──────────────────────────────────────────────────────────────
_lib = "/home/alireza/.idaes/lib_extracted/usr/lib/aarch64-linux-gnu"
os.environ["LD_LIBRARY_PATH"] = (
    f"{_lib}:{_lib}/blas:{_lib}/lapack:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)
BONMIN     = os.path.expanduser("~/.idaes/bin/bonmin")
Z_TARGET   = 1.3
TIME_LIMIT = 60    # seconds per solve; raise if any size times out

# ── 10 benchmark sizes ────────────────────────────────────────────────────────
# (n_continuous, k_integer, label)
# Calibrated from diagnostic runs so each size completes in < 30 s.
# k is kept ≤ 4 because B-OA's MILP master scales exponentially in k.
SIZES = [
    (    10,  1, "tiny"),
    (    50,  1, "small"),
    (   200,  2, "small+"),
    (   600,  2, "medium-"),
    (  1500,  2, "medium"),
    (  3000,  2, "medium+"),
    (  1500,  3, "large-"),
    (  3000,  3, "large"),
    (  3000,  4, "large+"),
    (  5000,  4, "xlarge"),
]


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(n: int, k: int) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel(name=f"MINLP_n{n}_k{k}")

    m.I = pyo.RangeSet(1, n)
    m.J = pyo.RangeSet(1, k)

    m.x = pyo.Var(m.I, domain=pyo.NonNegativeReals, bounds=(0, 3), initialize=1.0)
    m.z = pyo.Var(m.J, domain=pyo.Integers,         bounds=(0, 3), initialize=1)

    # Objective: pull x_i → 1 and z_j → 1.3 (nearest integer: 1)
    m.obj = pyo.Objective(
        expr=(
            pyo.quicksum((m.x[i] - 1) ** 2 for i in m.I)
            + pyo.quicksum((m.z[j] - Z_TARGET) ** 2 for j in m.J)
        ),
        sense=pyo.minimize,
    )

    # Separable non-linear constraints: x_i² + 0.5·z_{f(i)} ≤ 1.6
    # f(i) maps each x_i to exactly one z_j (evenly distributed)
    z_of = {i: (i - 1) * k // n + 1 for i in range(1, n + 1)}

    def ball_rule(model, i):
        return model.x[i] ** 2 + 0.5 * model.z[z_of[i]] <= 1.6

    m.c_ball = pyo.Constraint(m.I, rule=ball_rule)

    # Linear coupling: ensures z_j can't all be 0 when k is large
    m.c_sum = pyo.Constraint(
        expr=(
            pyo.quicksum(m.x[i] for i in m.I)
            + pyo.quicksum(m.z[j] for j in m.J)
            >= n + k - 1
        )
    )

    return m


# ── Single solve with timing ──────────────────────────────────────────────────

def run_one(n: int, k: int):
    t_build0 = time.perf_counter()
    m = build_model(n, k)
    t_build = time.perf_counter() - t_build0

    solver = pyo.SolverFactory("bonmin", executable=BONMIN)
    solver.options["bonmin.algorithm"]     = "B-OA"
    solver.options["bonmin.time_limit"]    = TIME_LIMIT
    solver.options["bonmin.bb_log_level"]  = 0
    solver.options["bonmin.nlp_log_level"] = 0
    solver.options["max_iter"]             = 100000

    t_solve0 = time.perf_counter()
    result = solver.solve(m, tee=False)
    t_solve = time.perf_counter() - t_solve0

    tc  = str(result.solver.termination_condition)
    obj = pyo.value(m.obj) if tc == "optimal" else float("nan")

    return dict(
        n=n, k=k,
        t_build_ms = t_build  * 1e3,
        t_solve_ms = t_solve  * 1e3,
        t_total_ms = (t_build + t_solve) * 1e3,
        obj        = obj,
        f_star     = (1 - Z_TARGET) ** 2 * k,  # 0.09 * k
        status     = tc,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    hdr = (
        f"{'size':>8}  {'n':>6} {'k':>3} {'vars':>6}  "
        f"{'build':>7} {'solve':>10} {'total':>10}   "
        f"{'objective':>10} {'f*':>6}   status"
    )
    sep = "─" * len(hdr)

    print()
    print("Scalable MINLP Benchmark  (Pyomo + Bonmin 1.8.8 · B-OA)")
    print("  min Σ(x_i−1)² + Σ(z_j−1.3)²")
    print("  s.t. x_i² + 0.5·z_{f(i)} ≤ 1.6   [separable non-linear, one per x_i]")
    print("       Σx_i + Σz_j ≥ n+k−1          [linear coupling]")
    print(f"  x_i∈[0,3] continuous,  z_j∈{{0..3}} integer,  f*=0.09·k")
    print(f"  Solver: Bonmin B-OA,  time limit: {TIME_LIMIT}s")
    print()
    print(hdr)
    print(sep)

    results = []
    for n, k, label in SIZES:
        r = run_one(n, k)
        r["label"] = label
        results.append(r)

        obj_str = f"{r['obj']:10.4f}" if r["obj"] == r["obj"] else "       ---"
        print(
            f"{label:>8}  {r['n']:>6} {r['k']:>3} {r['n']+r['k']:>6}  "
            f"{r['t_build_ms']:>6.1f}ms {r['t_solve_ms']:>9.1f}ms {r['t_total_ms']:>9.1f}ms   "
            f"{obj_str} {r['f_star']:>6.2f}   {r['status']}",
            flush=True,
        )

    print(sep)
    print("  n=continuous vars, k=integer vars, vars=n+k total")
    print("  build=Pyomo model construction, solve=Bonmin wall-clock")
    print("  f*=known optimum=0.09·k  (x_i=1, z_j=1 for all i,j)")

    solved = [r for r in results if r["status"] == "optimal"]
    if len(solved) >= 2:
        slowest = max(solved, key=lambda r: r["t_solve_ms"])
        fastest = min(solved, key=lambda r: r["t_solve_ms"])
        ratio = slowest["t_solve_ms"] / max(fastest["t_solve_ms"], 0.01)
        print(
            f"\n  Solve-time ratio  (slowest/fastest optimal): "
            f"{ratio:.0f}×  "
            f"({slowest['n']+slowest['k']} vars, k={slowest['k']} "
            f"vs {fastest['n']+fastest['k']} vars, k={fastest['k']})"
        )


if __name__ == "__main__":
    main()
