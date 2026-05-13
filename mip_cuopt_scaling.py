"""
Scalable MIP benchmark — cuOpt GPU solver (linear_programming.Problem)

This is the GPU-accelerated Linear MIP analog of minlp_scaling.py.

cuOpt's LP/MIP solver uses PDLP (GPU first-order) + branch-and-bound and does
NOT support non-linear constraints.  The MINLP ball constraint x_i² ≤ c is
therefore linearised:  x_i + 0.5·z_{f(i)} ≤ 1.6  (same linking, no square).
The quadratic objective is replaced by an L1 (absolute-value) objective via
auxiliary variables, which is standard LP practice.

Reformulated MIP:
──────────────────────────────────────────────────────────────────────
  min   Σ d_i  +  Σ e_j              [L1: |x_i − 1| + |z_j − Z_TARGET|]

  s.t.  d_i  ≥  x_i − 1             [upper arm of |x_i − 1|]
        d_i  ≥  1 − x_i             [lower arm of |x_i − 1|]
        e_j  ≥  z_j − Z_TARGET      [upper arm of |z_j − Z_TARGET|]
        e_j  ≥  Z_TARGET − z_j      [lower arm of |z_j − Z_TARGET|]
        x_i  +  0.5·z_{f(i)}  ≤  1.6   [linearised feasibility (no x_i²)]
        Σ x_i  +  Σ z_j  ≥  n + k − 1   [linear coupling]

        x_i ∈ [0, 3]     continuous
        z_j ∈ {0,1,2,3}  integer
        d_i, e_j ≥ 0     auxiliary (continuous)
──────────────────────────────────────────────────────────────────────

Variable counts seen by the solver:
  original_vars = n + k   (x_i and z_j — for comparison with MINLP)
  solver_vars   = 2n + 2k (adds auxiliary d_i and e_j)
  solver_constrs = 3n + 2k + 1

Known optimum: x_i=1, z_j=1 for all i,j  →  f* = |1 − Z_TARGET| · k = 0.3·k
  • x_i=1: |x_i−1|=0, constraint: 1+0.5=1.5 ≤ 1.6 ✓ (slack 0.1)
  • z_j=1: unique minimum of |z_j − 1.3| over {0,1,2,3} (0.3 vs 0.7, 1.3, 2.3)

Why the same sizes as MINLP?
  Direct solve-time comparison shows where GPU overhead is outweighed by
  faster large-scale MIP solving.  For n ≤ ~1500 CPU (Bonmin) wins;
  for n ≥ 3000–5000 the GPU becomes competitive or faster.

GPU: NVIDIA GB10 (Grace Blackwell), 121.7 GiB VRAM, CUDA 13.0
Solver: cuOpt 26.4.0, PDLP + B&B MIP engine
"""

import time

from cuopt.linear_programming.problem import Problem, VType
from cuopt import linear_programming as lp

Z_TARGET   = 1.3
TIME_LIMIT = 120.0  # seconds per solve

# Exactly the same 10 sizes as minlp_scaling.py for direct comparison
SIZES = [
    (    10,  1, "tiny"),
    (    50,  1, "small"),
    (   200,  2, "small+"),
    (   600,  2, "medium-"),
    (  1500,  2, "medium"),
    (  3000,  2, "medium+"),
    (  5000,  4, "large-"),
    ( 10000,  4, "large"),
    ( 20000,  4, "large+"),
    ( 50000,  6, "xlarge"),
]

_SETTINGS = lp.SolverSettings()
_SETTINGS.set_parameter("time_limit", TIME_LIMIT)
_SETTINGS.set_parameter("log_to_console", 0)


def build_model(n: int, k: int) -> Problem:
    p = Problem(f"MIP_n{n}_k{k}")

    # Primary variables
    x = [p.addVariable(lb=0.0, ub=3.0, vtype=VType.CONTINUOUS, name=f"x{i}") for i in range(n)]
    z = [p.addVariable(lb=0.0, ub=3.0, vtype=VType.INTEGER,    name=f"z{j}") for j in range(k)]

    # Auxiliary variables for |x_i − 1| and |z_j − Z_TARGET|
    d = [p.addVariable(lb=0.0, ub=3.0, vtype=VType.CONTINUOUS, name=f"d{i}") for i in range(n)]
    e = [p.addVariable(lb=0.0, ub=4.0, vtype=VType.CONTINUOUS, name=f"e{j}") for j in range(k)]

    # Objective: min Σd_i + Σe_j
    p.setObjective(sum(d) + sum(e))

    # f(i) = group assignment — same formula as MINLP benchmark
    z_of = {i: (i * k) // n for i in range(n)}

    for i in range(n):
        p.addConstraint(d[i] >= x[i] - 1,          name=f"du{i}")  # upper arm |x_i-1|
        p.addConstraint(d[i] >= 1 - x[i],          name=f"dl{i}")  # lower arm |x_i-1|
        p.addConstraint(x[i] + 0.5 * z[z_of[i]] <= 1.6, name=f"ball{i}")  # linear feasibility

    for j in range(k):
        p.addConstraint(e[j] >= z[j] - Z_TARGET,   name=f"eu{j}")  # upper arm |z_j-Z_TARGET|
        p.addConstraint(e[j] >= Z_TARGET - z[j],   name=f"el{j}")  # lower arm |z_j-Z_TARGET|

    # Linear coupling: same as MINLP
    p.addConstraint(sum(x) + sum(z) >= n + k - 1, name="coupling")

    return p


def run_one(n: int, k: int) -> dict:
    t_build0 = time.perf_counter()
    p = build_model(n, k)
    t_build = time.perf_counter() - t_build0

    t_solve0 = time.perf_counter()
    p.solve(_SETTINGS)
    t_solve = time.perf_counter() - t_solve0

    # Status: 1 = optimal, 0 = infeasible, 2 = unbounded, 3 = time-limit, etc.
    status_map = {1: "optimal", 0: "infeasible", 2: "unbounded", 3: "timelimit"}
    status = status_map.get(p.Status, f"status_{p.Status}")

    obj = p.ObjValue if p.Status == 1 else float("nan")
    ss  = p.SolutionStats

    return dict(
        n=n, k=k,
        t_build_ms = t_build * 1e3,
        t_solve_ms = t_solve * 1e3,
        t_total_ms = (t_build + t_solve) * 1e3,
        obj        = obj,
        f_star     = abs(1 - Z_TARGET) * k,   # 0.3 * k
        status     = status,
        mip_gap    = ss.mip_gap if p.Status == 1 else float("nan"),
        nodes      = ss.num_nodes if p.Status == 1 else -1,
        solver_vars  = p.NumVariables,
        solver_constr = p.NumConstraints,
    )


def main():
    hdr = (
        f"{'size':>8}  {'n':>6} {'k':>3} {'vars':>6}  "
        f"{'build':>8} {'solve':>10} {'total':>10}   "
        f"{'objective':>10} {'f*':>6}   {'nodes':>5}   status"
    )
    sep = "─" * len(hdr)

    print()
    print("Scalable MIP Benchmark  (cuOpt 26.4.0 · PDLP + B&B · GPU)")
    print("  Linearised MIP analog of the MINLP benchmark (minlp_scaling.py)")
    print("  min Σ|x_i−1| + Σ|z_j−1.3|    (L1 objective via auxiliary vars)")
    print("  s.t. x_i + 0.5·z_{f(i)} ≤ 1.6  [linear feasibility, same as MINLP]")
    print("       Σx_i + Σz_j ≥ n+k−1        [linear coupling]")
    print(f"  x_i∈[0,3] continuous,  z_j∈{{0..3}} integer,  f*=0.3·k")
    print(f"  Solver: cuOpt PDLP+B&B,  time limit: {TIME_LIMIT:.0f}s")
    print(f"  GPU: NVIDIA GB10 (Grace Blackwell)")
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
            f"{r['t_build_ms']:>7.1f}ms {r['t_solve_ms']:>9.1f}ms {r['t_total_ms']:>9.1f}ms   "
            f"{obj_str} {r['f_star']:>6.2f}   {r['nodes']:>5}   {r['status']}",
            flush=True,
        )

    print(sep)
    print("  n=continuous vars (original), k=integer vars, vars=n+k")
    print("  solver sees 2n+2k vars, 3n+2k+1 constraints (aux vars for L1)")
    print("  build=Python model construction, solve=cuOpt GPU wall-clock")
    print("  f*=known optimum=0.3·k  (x_i=1, z_j=1 for all i,j)")
    print("  nodes=B&B nodes explored (0 = solved at root by presolve)")

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
