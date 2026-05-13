"""
Simple Non-Linear Mixed-Integer Program (MINLP) solved with Pyomo + Bonmin.

Problem:
  Minimize    (x - 1)^2 + 2*(y - 1)^2 + (z - 1.5)^2

  Subject to:
    x^2 + y^2  <=  3 - 0.5*z      [non-linear: feasible ball shrinks as z grows]
    x  +  y  +  z  >=  2          [linear coupling constraint]

    0 <= x <= 3   (continuous)
    0 <= y <= 3   (continuous)
    z in {0, 1, 2, 3, 4}          (integer)

Intuition:
  The objective pulls toward (x, y, z) = (1, 1, 1.5). z=1 or z=2 minimises
  the (z-1.5)^2 term (both give 0.25). With z=1 the ball radius^2 = 2.5
  (loose), while z=2 makes it exactly 2.0 (tight at x=y=1). Both yield the
  same objective value of 0.25, so the solver may return either.
"""

import os
import time
import pyomo.environ as pyo

# ── Library path so Bonmin can find libgfortran / BLAS / LAPACK ──────────────
_lib = "/home/alireza/.idaes/lib_extracted/usr/lib/aarch64-linux-gnu"
os.environ["LD_LIBRARY_PATH"] = (
    f"{_lib}:{_lib}/blas:{_lib}/lapack:"
    + os.environ.get("LD_LIBRARY_PATH", "")
)

BONMIN = os.path.expanduser("~/.idaes/bin/bonmin")

# ── Model ────────────────────────────────────────────────────────────────────

def build_model():
    m = pyo.ConcreteModel(name="Simple MINLP")

    m.x = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 3), initialize=1.0)
    m.y = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0, 3), initialize=1.0)
    m.z = pyo.Var(domain=pyo.Integers,         bounds=(0, 4), initialize=1)

    m.obj = pyo.Objective(
        expr=(m.x - 1)**2 + 2*(m.y - 1)**2 + (m.z - 1.5)**2,
        sense=pyo.minimize,
    )

    # Non-linear: x^2 + y^2 <= 3 - 0.5*z
    m.c_ball = pyo.Constraint(expr=m.x**2 + m.y**2 <= 3 - 0.5 * m.z)

    # Linear coupling
    m.c_sum  = pyo.Constraint(expr=m.x + m.y + m.z >= 2)

    return m


# ── Solve ────────────────────────────────────────────────────────────────────

def solve(model, tee=False):
    solver = pyo.SolverFactory("bonmin", executable=BONMIN)
    solver.options["bonmin.bb_log_level"]   = 0   # suppress branch-and-bound log
    solver.options["bonmin.nlp_log_level"]  = 0   # suppress NLP sub-solver log

    t0 = time.perf_counter()
    result = solver.solve(model, tee=tee)
    elapsed = time.perf_counter() - t0

    return result, elapsed


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    m = build_model()

    print("=" * 50)
    print("MINLP Problem")
    print("=" * 50)
    print("  min  (x-1)² + 2(y-1)² + (z-1.5)²")
    print("  s.t. x² + y²  ≤  3 - 0.5z")
    print("       x + y + z ≥  2")
    print("       0 ≤ x,y ≤ 3  (continuous)")
    print("       z ∈ {0,1,2,3,4}  (integer)")
    print()

    result, elapsed = solve(m)

    status     = result.solver.status
    termcond   = result.solver.termination_condition

    print(f"Solver status           : {status}")
    print(f"Termination condition   : {termcond}")
    print()
    print(f"  x  = {pyo.value(m.x):.6f}")
    print(f"  y  = {pyo.value(m.y):.6f}")
    print(f"  z  = {pyo.value(m.z):.0f}")
    print(f"  Objective = {pyo.value(m.obj):.6f}")
    print()
    print(f"  c_ball slack  : {3 - 0.5*pyo.value(m.z) - pyo.value(m.x)**2 - pyo.value(m.y)**2:+.6f}")
    print(f"  c_sum  slack  : {pyo.value(m.x) + pyo.value(m.y) + pyo.value(m.z) - 2:+.6f}")
    print()
    print(f"Wall-clock time         : {elapsed*1000:.3f} ms")
