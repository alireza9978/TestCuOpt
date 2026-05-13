"""
Hybrid B-OA Solver: IPOPT (NLP sub-problems) + cuOpt GPU (MILP master)

Bonmin's B-OA algorithm alternates between two sub-solvers:
  NLP step  → IPOPT  (relaxes integer constraints, solves continuous NLP)
  MILP step → CBC    (solves linearised master problem with integer variables)

This script replaces CBC with cuOpt, running the MILP master on the GPU.
Everything else — the OA algorithm, IPOPT for NLPs — stays identical to Bonmin.

Problem (from minlp_pyomo.py):
  min  (x−1)² + 2(y−1)² + (z−1.5)²
  s.t. x² + y²  ≤  3 − 0.5z        [non-linear ball constraint]
       x  + y  + z  ≥  2            [linear coupling]
       x, y ∈ [0, 3]  continuous
       z ∈ {0, 1, 2, 3, 4}  integer

B-OA algorithm (Duran & Grossmann, 1986):
  Phase 0: Solve NLP relaxation (z continuous) → initial OA cuts.
  Loop:
    Step A (MILP master, cuOpt):
      Solve linear MIP with all accumulated OA cuts.
      The epigraph variable η lower-bounds the true objective.
      → LB = η*,  candidate integers ẑ
    Step B (NLP fix, IPOPT):
      Fix z = ẑ, solve the continuous NLP.
      → UB = min(UB, f(x̄, ȳ, ẑ))
    Step C (Cut generation):
      Add objective tangent cut at (x̄, ȳ, ẑ):
        η ≥ 2(x̄−1)·x + 4(ȳ−1)·y + 2(ẑ−1.5)·z + constant
      Add ball tangent cut at (x̄, ȳ):
        2x̄·x + 2ȳ·y + 0.5z ≤ 3 + x̄² + ȳ²
    Step D (Convergence):
      If (UB − LB) / max(1, |UB|) < tol → global optimum found.
"""

import math
import os
import time

import pyomo.environ as pyo

from cuopt.linear_programming.problem import Problem, VType
from cuopt import linear_programming as lp

# ── Library / solver paths ───────────────────────────────────────────────────
_lib = "/home/alireza/.idaes/lib_extracted/usr/lib/aarch64-linux-gnu"
os.environ["LD_LIBRARY_PATH"] = (
    f"{_lib}:{_lib}/blas:{_lib}/lapack:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)
IPOPT  = os.path.expanduser("~/.idaes/bin/ipopt")
BONMIN = os.path.expanduser("~/.idaes/bin/bonmin")

# ── Algorithm parameters ──────────────────────────────────────────────────────
TOL      = 1e-5   # relative gap tolerance for convergence
MAX_ITER = 20     # safety limit on outer iterations

# ── cuOpt settings (created once, reused every MILP solve) ───────────────────
_cuopt_settings = lp.SolverSettings()
_cuopt_settings.set_parameter("time_limit",     60.0)
_cuopt_settings.set_parameter("log_to_console", 0)

# ── IPOPT solver factory (created once) ──────────────────────────────────────
_ipopt = pyo.SolverFactory("ipopt", executable=IPOPT)
_ipopt.options["print_level"] = 0
_ipopt.options["max_iter"]    = 3000
_ipopt.options["tol"]         = 1e-9


# ── NLP sub-problems ─────────────────────────────────────────────────────────

def solve_nlp_relaxation():
    """NLP relaxation: z is treated as a continuous variable in [0, 4].
    This is the first step of B-OA and provides the initial OA cut point.
    Returns (x, y, z, obj) or None if solver fails."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 3), initialize=1.0)
    m.y = pyo.Var(bounds=(0, 3), initialize=1.0)
    m.z = pyo.Var(bounds=(0, 4), initialize=1.5)   # continuous relaxation

    m.obj    = pyo.Objective(expr=(m.x-1)**2 + 2*(m.y-1)**2 + (m.z-1.5)**2)
    m.c_ball = pyo.Constraint(expr=m.x**2 + m.y**2 + 0.5*m.z <= 3)
    m.c_sum  = pyo.Constraint(expr=m.x + m.y + m.z >= 2)

    res = _ipopt.solve(m, tee=False)
    tc  = str(res.solver.termination_condition)
    if tc in ("optimal", "locallyOptimal"):
        return pyo.value(m.x), pyo.value(m.y), pyo.value(m.z), pyo.value(m.obj)
    return None


def solve_nlp_fixed(z_val: int):
    """NLP fix: solve the NLP with z pinned to a specific integer value.
    Returns (x, y, obj, feasible)."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 3), initialize=1.0)
    m.y = pyo.Var(bounds=(0, 3), initialize=1.0)

    m.obj    = pyo.Objective(expr=(m.x-1)**2 + 2*(m.y-1)**2 + (z_val-1.5)**2)
    m.c_ball = pyo.Constraint(expr=m.x**2 + m.y**2 + 0.5*z_val <= 3)
    m.c_sum  = pyo.Constraint(expr=m.x + m.y + z_val >= 2)

    res = _ipopt.solve(m, tee=False)
    tc  = str(res.solver.termination_condition)
    if tc in ("optimal", "locallyOptimal"):
        return pyo.value(m.x), pyo.value(m.y), pyo.value(m.obj), True
    return None, None, None, False


# ── MILP master (cuOpt GPU) ───────────────────────────────────────────────────

def solve_milp_master(obj_cuts: list, ball_cuts: list):
    """Build and solve the MILP master problem with cuOpt.

    Variables:
      x, y ∈ [0, 3]   continuous
      z    ∈ {0,..,4}  integer
      η    ∈ ℝ         epigraph variable (lower-bounds the true objective)

    Constraints:
      Objective tangent cuts at each previous NLP solution:
        η ≥ 2(x₀−1)·x + 4(y₀−1)·y + 2(z₀−1.5)·z + c
      Ball tangent cuts at each previous NLP solution:
        2x₀·x + 2y₀·y + 0.5z ≤ 3 + x₀² + y₀²
      Original linear coupling (unchanged):
        x + y + z ≥ 2

    obj_cuts : list of (x0, y0, z0) — NLP solution points for objective cuts
    ball_cuts: list of (x0, y0)      — NLP solution points for ball cuts
    Returns (x, y, z_int, eta) or (None, None, None, None) if infeasible.
    """
    p   = Problem("MILP_master")
    x   = p.addVariable(lb=0.0, ub=3.0, vtype=VType.CONTINUOUS, name="x")
    y   = p.addVariable(lb=0.0, ub=3.0, vtype=VType.CONTINUOUS, name="y")
    z   = p.addVariable(lb=0.0, ub=4.0, vtype=VType.INTEGER,    name="z")
    eta = p.addVariable(lb=0.0, ub=1e4, vtype=VType.CONTINUOUS, name="eta")

    p.setObjective(eta)

    # ── Objective tangent cuts ────────────────────────────────────────────────
    # f(x,y,z) = (x−1)² + 2(y−1)² + (z−1.5)²
    # ∇f(x₀,y₀,z₀) = (2(x₀−1), 4(y₀−1), 2(z₀−1.5))
    # η ≥ f₀ + ∇f·(w−w₀)
    # η ≥ 2(x₀−1)·x + 4(y₀−1)·y + 2(z₀−1.5)·z + [5.25 − x₀² − 2y₀² − z₀²]
    #   (constant term derived by expanding and collecting: see optimization_report.md §5)
    for (x0, y0, z0) in obj_cuts:
        coef_x = 2 * (x0 - 1)
        coef_y = 4 * (y0 - 1)
        coef_z = 2 * (z0 - 1.5)
        const  = 5.25 - x0**2 - 2*y0**2 - z0**2
        p.addConstraint(eta >= coef_x*x + coef_y*y + coef_z*z + const,
                        name=f"obj_cut_{len(p._constraints) if hasattr(p,'_constraints') else ''}")

    # ── Ball tangent cuts ─────────────────────────────────────────────────────
    # g(x,y,z) = x²+y²+0.5z,  ∇g = (2x₀, 2y₀, 0.5)
    # Tangent: 2x₀·x + 2y₀·y + 0.5z ≤ 3 + x₀² + y₀²
    for (x0, y0) in ball_cuts:
        rhs = 3.0 + x0**2 + y0**2
        p.addConstraint(2*x0*x + 2*y0*y + 0.5*z <= rhs)

    # ── Original linear coupling ──────────────────────────────────────────────
    p.addConstraint(x + y + z >= 2, name="coupling")

    p.solve(_cuopt_settings)

    if p.Status == 1:   # optimal
        return x.Value, y.Value, round(z.Value), eta.Value
    return None, None, None, None


# ── Bonmin reference solve ────────────────────────────────────────────────────

def solve_bonmin():
    """Solve the same problem with pure Bonmin (B-OA + CBC) for comparison."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0,3), initialize=1.0)
    m.y = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0,3), initialize=1.0)
    m.z = pyo.Var(domain=pyo.Integers,         bounds=(0,4), initialize=1)
    m.obj    = pyo.Objective(expr=(m.x-1)**2 + 2*(m.y-1)**2 + (m.z-1.5)**2)
    m.c_ball = pyo.Constraint(expr=m.x**2 + m.y**2 <= 3 - 0.5*m.z)
    m.c_sum  = pyo.Constraint(expr=m.x + m.y + m.z >= 2)

    solver = pyo.SolverFactory("bonmin", executable=BONMIN)
    solver.options["bonmin.algorithm"]     = "B-OA"
    solver.options["bonmin.bb_log_level"]  = 0
    solver.options["bonmin.nlp_log_level"] = 0

    t0 = time.perf_counter()
    res = solver.solve(m, tee=False)
    elapsed = time.perf_counter() - t0
    tc = str(res.solver.termination_condition)
    obj = pyo.value(m.obj) if tc == "optimal" else float("nan")
    return pyo.value(m.x), pyo.value(m.y), int(round(pyo.value(m.z))), obj, elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    W = 62
    print("═" * W)
    print("Hybrid B-OA:  IPOPT (NLP)  +  cuOpt GPU (MILP master)")
    print("═" * W)
    print("  Problem:  min  (x−1)² + 2(y−1)² + (z−1.5)²")
    print("    s.t.  x²+y² ≤ 3−0.5z,   x+y+z ≥ 2")
    print("    x,y∈[0,3] continuous,   z∈{0..4} integer")
    print()
    print("  NLP sub-solver : IPOPT (via Pyomo)")
    print("  MILP sub-solver: cuOpt 26.4.0 (PDLP+B&B, GPU)")
    print("  Convergence tol: {:.0e}  |  Max iterations: {}".format(TOL, MAX_ITER))
    print()

    obj_cuts  = []   # (x0, y0, z0) — for objective epigraph cuts
    ball_cuts = []   # (x0, y0)      — for ball tangent cuts
    UB        = float("inf")
    best      = None
    t_total   = time.perf_counter()
    t_nlp_total  = 0.0
    t_milp_total = 0.0

    # ── Phase 0: NLP relaxation ───────────────────────────────────────────────
    print("─" * W)
    t0 = time.perf_counter()
    relax = solve_nlp_relaxation()
    t_nlp_total += time.perf_counter() - t0

    if relax is None:
        print("  NLP relaxation failed — cannot proceed.")
        return

    xr, yr, zr, fr = relax
    obj_cuts.append((xr, yr, zr))
    ball_cuts.append((xr, yr))
    print(f"  Phase 0 — NLP relaxation (z continuous, solved by IPOPT):")
    print(f"    x={xr:.4f}  y={yr:.4f}  z={zr:.4f}  f={fr:.6f}")
    print(f"    → Added 1 objective cut + 1 ball cut")
    print()

    # ── Main B-OA loop ────────────────────────────────────────────────────────
    for it in range(1, MAX_ITER + 1):
        print(f"  Iter {it:2d} {'─'*(W-9)}")

        # Step A: MILP master (cuOpt)
        t0 = time.perf_counter()
        xm, ym, zm, eta_m = solve_milp_master(obj_cuts, ball_cuts)
        t_milp = time.perf_counter() - t0
        t_milp_total += t_milp

        if xm is None:
            print("    MILP infeasible — stopping.")
            break

        LB = eta_m
        print(f"    MILP (cuOpt, {t_milp*1e3:5.0f} ms): "
              f"z={zm}  x={xm:.4f}  y={ym:.4f}  η={eta_m:.6f}   LB={LB:.6f}")

        # Step B: NLP fix (IPOPT)
        t0 = time.perf_counter()
        xn, yn, fn, feasible = solve_nlp_fixed(zm)
        t_nlp = time.perf_counter() - t0
        t_nlp_total += t_nlp

        if feasible:
            if fn < UB - 1e-10:
                UB   = fn
                best = (xn, yn, zm, fn)
            obj_cuts.append((xn, yn, float(zm)))
            ball_cuts.append((xn, yn))
            ball_slack    = (3 - 0.5*zm) - (xn**2 + yn**2)
            coupling_slack = xn + yn + zm - 2
            print(f"    NLP  (IPOPT, {t_nlp*1e3:5.0f} ms): "
                  f"z={zm}  x={xn:.4f}  y={yn:.4f}  f={fn:.6f}   UB={UB:.6f}")
            print(f"      ball slack={ball_slack:+.4f}  coupling slack={coupling_slack:+.4f}")
        else:
            print(f"    NLP  (IPOPT, {t_nlp*1e3:5.0f} ms): z={zm} → INFEASIBLE "
                  f"(ball constraint too tight); no UB update")

        # Step D: convergence
        if UB < float("inf"):
            gap = abs(UB - LB) / max(1.0, abs(UB))
        else:
            gap = float("inf")

        print(f"    Gap: |{UB:.6f} − {LB:.6f}| / {max(1.0,abs(UB)):.6f} = {gap:.2e}", end="")

        if gap <= TOL:
            print("  ✓ CONVERGED")
            break
        else:
            print(f"  (tol={TOL:.0e})")
            print(f"    Cuts pool: {len(obj_cuts)} obj cuts, {len(ball_cuts)} ball cuts")
        print()

    t_total = time.perf_counter() - t_total

    # ── Results ───────────────────────────────────────────────────────────────
    print()
    print("═" * W)
    print("  HYBRID B-OA RESULT")
    print("═" * W)
    if best:
        xb, yb, zb, fb = best
        ball_slack_b    = (3 - 0.5*zb) - (xb**2 + yb**2)
        coupling_slack_b = xb + yb + zb - 2
        print(f"  x = {xb:.6f}")
        print(f"  y = {yb:.6f}")
        print(f"  z = {zb}")
        print(f"  Objective (L2) = {fb:.6f}")
        print()
        print(f"  Ball slack  (3 − 0.5z − x² − y²) : {ball_slack_b:+.6f}")
        print(f"  Coupling slack (x+y+z − 2)        : {coupling_slack_b:+.6f}")
    else:
        print("  No feasible solution found.")

    print()
    print(f"  Wall-clock time breakdown:")
    print(f"    IPOPT (NLP sub-problems)  : {t_nlp_total*1e3:7.1f} ms")
    print(f"    cuOpt (MILP master, GPU)  : {t_milp_total*1e3:7.1f} ms")
    print(f"    Total (hybrid B-OA)       : {t_total*1e3:7.1f} ms")

    # ── Bonmin reference ──────────────────────────────────────────────────────
    print()
    print("─" * W)
    print("  Running Bonmin B-OA (IPOPT + CBC, pure CPU) for comparison …")
    xbm, ybm, zbm, fbm, t_bonmin = solve_bonmin()
    print(f"    x={xbm:.6f}  y={ybm:.6f}  z={zbm}  obj={fbm:.6f}")
    print(f"    Wall-clock: {t_bonmin*1e3:.1f} ms")

    print()
    print("─" * W)
    print("  Side-by-side comparison:")
    print(f"  {'':22s}  {'Hybrid (IPOPT+cuOpt)':>22s}  {'Bonmin (IPOPT+CBC)':>20s}")
    print(f"  {'x':22s}  {xb if best else float('nan'):>22.6f}  {xbm:>20.6f}")
    print(f"  {'y':22s}  {yb if best else float('nan'):>22.6f}  {ybm:>20.6f}")
    print(f"  {'z':22s}  {zb if best else -1:>22d}  {zbm:>20d}")
    print(f"  {'objective':22s}  {fb if best else float('nan'):>22.6f}  {fbm:>20.6f}")
    print(f"  {'solve time':22s}  {t_total*1e3:>21.1f}ms  {t_bonmin*1e3:>19.1f}ms")
    print()
    if best and abs(fb - fbm) < 1e-4:
        print("  ✓ Solutions match — cuOpt GPU gives the same optimal answer as CBC.")
    else:
        print("  ✗ Solutions differ — check algorithm.")
    print("═" * W)


if __name__ == "__main__":
    main()
