"""
Scalable Hybrid B-OA Benchmark: IPOPT (NLP sub-problems) + cuOpt GPU (MILP master)

Runs the same 10 benchmark sizes as minlp_scaling.py, replacing Bonmin's CBC
MILP sub-solver with cuOpt (GPU).

Problem at size (n, k):
  min   Σ_{i=1..n} (x_i − 1)²  +  Σ_{j=1..k} (z_j − 1.3)²
  s.t.  x_i² + 0.5·z_{f(i)}  ≤  1.6      [non-linear ball constraint]
        Σ x_i  +  Σ z_j        ≥  n+k−1   [linear coupling]
        x_i ∈ [0, 3]   continuous
        z_j ∈ {0,1,2,3}  integer

B-OA iteration:
  Phase 0: NLP relaxation (all variables continuous) → initial OA cuts
  Each iteration:
    A. MILP master (cuOpt GPU): minimise epigraph variable η subject to
         - T objective tangent cuts  η ≥ Σ 2(x̄_i−1)·x_i + Σ 2(z̄_j−1.3)·z_j + const_t
         - T·n ball tangent cuts     2x̄_i·x_i + 0.5·z_{f(i)} ≤ 1.6 + x̄_i²
         - coupling                  Σ x_i + Σ z_j ≥ n+k−1
    B. NLP fix (IPOPT): fix z to integer vector from MILP, solve continuous NLP → UB
    C. Add new OA cuts from NLP fix point
    D. Convergence: (UB − LB) / max(1, |UB|) < 1e-5

Known global optimum: x_i=1, z_j=1 → f* = k·(1−1.3)² = 0.09·k
"""

import math as _math
import os
import time

import pyomo.environ as pyo

from cuopt.linear_programming.problem import Problem, VType
from cuopt import linear_programming as lp

# ── Paths ─────────────────────────────────────────────────────────────────────
_lib = "/home/alireza/.idaes/lib_extracted/usr/lib/aarch64-linux-gnu"
os.environ["LD_LIBRARY_PATH"] = (
    f"{_lib}:{_lib}/blas:{_lib}/lapack:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)
IPOPT  = os.path.expanduser("~/.idaes/bin/ipopt")
BONMIN = os.path.expanduser("~/.idaes/bin/bonmin")

# ── Problem parameters ────────────────────────────────────────────────────────
Z_TARGET = 1.3
RHS      = 1.6   # ball constraint RHS: x_i² + 0.5·z_{f(i)} ≤ 1.6

# ── Algorithm parameters ──────────────────────────────────────────────────────
TOL      = 1e-3   # relative gap; PDLP accuracy limits sub-1e-4 gaps at large n
MAX_ITER = 6

# ── Solver settings ───────────────────────────────────────────────────────────
_cuopt_settings = lp.SolverSettings()
_cuopt_settings.set_parameter("time_limit",     300.0)
_cuopt_settings.set_parameter("log_to_console", 0)

_ipopt = pyo.SolverFactory("ipopt", executable=IPOPT)
_ipopt.options["print_level"] = 0
_ipopt.options["max_iter"]    = 3000
_ipopt.options["tol"]         = 1e-9

# ── Benchmark sizes ───────────────────────────────────────────────────────────
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


def z_of(i, n, k):
    """Group assignment (0-indexed): same formula as minlp_scaling.py."""
    return (i * k) // n


# ── NLP sub-problems ──────────────────────────────────────────────────────────

def build_nlp_relaxation(n, k):
    """Build Pyomo NLP relaxation model (z continuous)."""
    zo = {i: z_of(i, n, k) for i in range(n)}

    m = pyo.ConcreteModel()
    m.x = pyo.Var(range(n), bounds=(0.0, 3.0), initialize=1.0)
    m.z = pyo.Var(range(k), bounds=(0.0, 3.0), initialize=Z_TARGET)

    m.obj = pyo.Objective(
        expr=(sum((m.x[i] - 1)**2 for i in range(n))
              + sum((m.z[j] - Z_TARGET)**2 for j in range(k)))
    )
    m.c_ball = pyo.ConstraintList()
    for i in range(n):
        m.c_ball.add(m.x[i]**2 + 0.5 * m.z[zo[i]] <= RHS)
    m.c_coupling = pyo.Constraint(
        expr=sum(m.x[i] for i in range(n)) + sum(m.z[j] for j in range(k)) >= n + k - 1
    )
    return m, zo


def solve_nlp_relaxation(n, k):
    """Solve NLP relaxation; returns (x_list, z_list, obj) or None."""
    m, _ = build_nlp_relaxation(n, k)
    res  = _ipopt.solve(m, tee=False)
    tc   = str(res.solver.termination_condition)
    if tc in ("optimal", "locallyOptimal"):
        return ([pyo.value(m.x[i]) for i in range(n)],
                [pyo.value(m.z[j]) for j in range(k)],
                pyo.value(m.obj))
    return None


def solve_nlp_fixed(n, k, z_vals):
    """NLP fix: z pinned to integer vector z_vals; returns (x_list, obj, feasible)."""
    zo = {i: z_of(i, n, k) for i in range(n)}

    m = pyo.ConcreteModel()
    m.x = pyo.Var(range(n), bounds=(0.0, 3.0), initialize=1.0)

    m.obj = pyo.Objective(
        expr=(sum((m.x[i] - 1)**2 for i in range(n))
              + sum((z_vals[j] - Z_TARGET)**2 for j in range(k)))
    )
    m.c_ball = pyo.ConstraintList()
    for i in range(n):
        m.c_ball.add(m.x[i]**2 + 0.5 * z_vals[zo[i]] <= RHS)
    m.c_coupling = pyo.Constraint(
        expr=(sum(m.x[i] for i in range(n))
              + sum(z_vals[j] for j in range(k)) >= n + k - 1)
    )

    res = _ipopt.solve(m, tee=False)
    tc  = str(res.solver.termination_condition)
    if tc in ("optimal", "locallyOptimal"):
        return ([pyo.value(m.x[i]) for i in range(n)],
                pyo.value(m.obj), True)
    return None, None, False


# ── MILP master (cuOpt GPU) ───────────────────────────────────────────────────

def solve_milp_master(n, k, obj_cuts, ball_cut_xs):
    """Build and solve the MILP master with cuOpt.

    obj_cuts     : list of (x0_list, z0_list) — one entry per past NLP solution
    ball_cut_xs  : list of x0_list            — one entry per past NLP solution

    Objective tangent cut at (x̄, z̄):
      η ≥ Σ 2(x̄_i−1)·x_i + Σ 2(z̄_j−Z_TARGET)·z_j + const
      const = n − Σx̄_i² + k·Z_TARGET² − Σz̄_j²

    Ball tangent cut at x̄_i:
      2x̄_i·x_i + 0.5·z_{f(i)} ≤ RHS + x̄_i²

    Returns (x_vals, z_int_vals, eta, t_build_s, t_solve_s)
    or (None, None, None, t_build_s, t_solve_s) if infeasible.
    """
    zo = {i: z_of(i, n, k) for i in range(n)}

    t_b0 = time.perf_counter()
    p   = Problem(f"MILP_n{n}_k{k}")
    x   = [p.addVariable(lb=0.0, ub=3.0, vtype=VType.CONTINUOUS, name=f"x{i}")
           for i in range(n)]
    z   = [p.addVariable(lb=0.0, ub=3.0, vtype=VType.INTEGER,    name=f"z{j}")
           for j in range(k)]
    eta = p.addVariable(lb=0.0, ub=1e6,  vtype=VType.CONTINUOUS, name="eta")

    p.setObjective(eta)

    # Objective tangent cuts (one per NLP solution; dense n+k coefficient row)
    for t, (x0, z0) in enumerate(obj_cuts):
        lhs   = eta
        lhs   = lhs - sum((2 * (x0[i] - 1)) * x[i] for i in range(n))
        lhs   = lhs - sum((2 * (z0[j] - Z_TARGET)) * z[j] for j in range(k))
        const = (n - sum(x0[i]**2 for i in range(n))
                 + k * Z_TARGET**2 - sum(z0[j]**2 for j in range(k)))
        p.addConstraint(lhs >= const, name=f"obj_cut_{t}")

    # Ball tangent cuts (one per x_i per NLP solution)
    for t, x0 in enumerate(ball_cut_xs):
        for i in range(n):
            rhs_i = RHS + x0[i]**2
            p.addConstraint(2 * x0[i] * x[i] + 0.5 * z[zo[i]] <= rhs_i,
                            name=f"ball_{t}_{i}")

    # Linear coupling
    p.addConstraint(sum(x) + sum(z) >= n + k - 1, name="coupling")

    t_build = time.perf_counter() - t_b0

    t_s0 = time.perf_counter()
    p.solve(_cuopt_settings)
    t_solve = time.perf_counter() - t_s0

    if p.Status == 1:
        xv = [x[i].Value for i in range(n)]
        zv = [round(z[j].Value) for j in range(k)]
        return xv, zv, eta.Value, t_build, t_solve
    return None, None, None, t_build, t_solve


# ── Run one benchmark size ────────────────────────────────────────────────────

def run_one(n, k, label):
    t_wall = time.perf_counter()

    obj_cuts    = []   # list of (x0_list, z0_list)
    ball_cut_xs = []   # list of x0_list
    UB          = float("inf")
    LB          = float("-inf")
    best_obj    = None
    t_nlp_ms    = 0.0
    t_build_ms  = 0.0
    t_solve_ms  = 0.0
    niters      = 0
    conv_status = "not_converged"

    # ── Phase 0: NLP relaxation ───────────────────────────────────────────────
    t0 = time.perf_counter()
    relax = solve_nlp_relaxation(n, k)
    t_nlp_ms += (time.perf_counter() - t0) * 1e3

    if relax is None:
        return {"label": label, "n": n, "k": k, "status": "nlp_relax_failed",
                "niters": 0, "t_total_ms": (time.perf_counter() - t_wall) * 1e3}

    xr, zr, fr = relax
    obj_cuts.append((xr, zr))
    ball_cut_xs.append(xr)

    # ── B-OA loop ─────────────────────────────────────────────────────────────
    for it in range(1, MAX_ITER + 1):
        niters = it

        # Step A: MILP master (cuOpt)
        xm, zm, eta_m, tb, ts = solve_milp_master(n, k, obj_cuts, ball_cut_xs)
        t_build_ms += tb * 1e3
        t_solve_ms += ts * 1e3

        if xm is None:
            conv_status = "milp_infeasible"
            break

        LB = eta_m

        # Step B: NLP fix (IPOPT)
        t0 = time.perf_counter()
        xn, fn, feasible = solve_nlp_fixed(n, k, zm)
        t_nlp_ms += (time.perf_counter() - t0) * 1e3

        if feasible and fn is not None:
            if best_obj is None or fn < best_obj - 1e-10:
                best_obj = fn
                UB       = fn
            obj_cuts.append((xn, [float(v) for v in zm]))
            ball_cut_xs.append(xn)
        else:
            # NLP infeasibility cut: add ball cuts at max feasible x̂_i for z=zm.
            # x̂_i = sqrt(max(0, RHS - 0.5·z_j)) is the tightest valid OA point.
            zo_local = {i: z_of(i, n, k) for i in range(n)}
            x_hat = [_math.sqrt(max(0.0, RHS - 0.5 * zm[zo_local[i]])) for i in range(n)]
            ball_cut_xs.append(x_hat)

        # Step D: convergence
        gap = (abs(UB - LB) / max(1.0, abs(UB))
               if UB < float("inf") else float("inf"))

        if gap <= TOL:
            conv_status = "optimal"
            break

    t_total_ms = (time.perf_counter() - t_wall) * 1e3

    f_star = k * (1.0 - Z_TARGET) ** 2   # = 0.09 · k

    return {
        "label":        label,
        "n":            n,
        "k":            k,
        "niters":       niters,
        "t_nlp_ms":     t_nlp_ms,
        "t_build_ms":   t_build_ms,
        "t_solve_ms":   t_solve_ms,
        "t_total_ms":   t_total_ms,
        "objective":    best_obj if best_obj is not None else float("nan"),
        "f_star":       f_star,
        "LB":           LB,
        "UB":           UB,
        "status":       conv_status,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    hdr = (
        f"{'size':>8}  {'n':>6} {'k':>3} {'iters':>5}  "
        f"{'nlp':>8} {'milp-build':>11} {'milp-solve':>11} {'total':>10}   "
        f"{'objective':>10} {'f*':>6}   status"
    )
    sep = "─" * len(hdr)

    print()
    print("Scalable Hybrid B-OA  (IPOPT NLP  +  cuOpt 26.4.0 GPU MILP)")
    print("  min Σ(x_i−1)² + Σ(z_j−1.3)²,   Z*=1.3,  f*=0.09·k")
    print("  s.t. x_i² + 0.5·z_{f(i)} ≤ 1.6  (non-linear)")
    print("       Σx_i + Σz_j ≥ n+k−1")
    print(f"  NLP: IPOPT,  MILP: cuOpt PDLP+B&B,  tol={TOL:.0e},  max_iter={MAX_ITER}")
    print(f"  GPU: NVIDIA GB10 (Grace Blackwell)")
    print()
    print(hdr)
    print(sep)

    results = []
    for n, k, label in SIZES:
        r = run_one(n, k, label)
        results.append(r)

        obj_str = f"{r['objective']:10.4f}" if r.get("objective") == r.get("objective") else "       ---"
        fstar   = r.get("f_star", float("nan"))
        total_s = r["t_total_ms"]
        print(
            f"{label:>8}  {r['n']:>6} {r['k']:>3} {r.get('niters',0):>5}  "
            f"{r.get('t_nlp_ms',0):>7.0f}ms "
            f"{r.get('t_build_ms',0):>10.0f}ms "
            f"{r.get('t_solve_ms',0):>10.0f}ms "
            f"{total_s:>9.0f}ms   "
            f"{obj_str} {fstar:>6.4f}   {r['status']}",
            flush=True,
        )

    print(sep)
    print("  iters=B-OA outer iterations (phase-0 NLP + iters MILP-NLP pairs)")
    print("  nlp=IPOPT time (all NLP calls),  milp-build=cuOpt model Python build,")
    print("  milp-solve=cuOpt GPU wall-clock,  total=end-to-end")
    print("  f*=0.09·k  (x_i=1, z_j=1 globally optimal)")

    solved = [r for r in results if r.get("status") == "optimal"]
    print(f"\n  Solved optimally: {len(solved)}/{len(results)}")
    if solved:
        slowest = max(solved, key=lambda r: r["t_total_ms"])
        fastest = min(solved, key=lambda r: r["t_total_ms"])
        if fastest["t_total_ms"] > 0:
            ratio = slowest["t_total_ms"] / fastest["t_total_ms"]
            print(f"  Total-time ratio (slowest/fastest optimal): {ratio:.0f}×  "
                  f"({slowest['label']}: {slowest['t_total_ms']:.0f}ms  vs  "
                  f"{fastest['label']}: {fastest['t_total_ms']:.0f}ms)")


if __name__ == "__main__":
    main()
