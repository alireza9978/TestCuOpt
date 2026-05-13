# Scalable Hybrid B-OA Benchmark Results

**Algorithm:** Custom B-OA (Outer Approximation) — IPOPT (NLP sub-problems) + cuOpt GPU (MILP master)  
**Hardware:** NVIDIA GB10 (Grace Blackwell), 121.7 GiB VRAM, CUDA 13.0, Ubuntu 24.04 LTS, aarch64  
**NLP solver:** IPOPT 3.13.2 (via IDAES, OpenMP-enabled)  
**MILP solver:** cuOpt 26.4.0 (PDLP + Branch-and-Bound, GPU)  
**Date:** 2026-05-13  
**Companion benchmarks:** `minlp_scaling_results.md` (Bonmin CPU), `mip_cuopt_scaling_results.md` (direct cuOpt MIP)

---

## What is the Hybrid B-OA?

Bonmin's B-OA (Outer Approximation) algorithm alternates between two sub-solvers:

| Sub-problem | Standard Bonmin | Hybrid (this benchmark) |
|---|---|---|
| **NLP relaxation / NLP fix** | IPOPT | IPOPT (unchanged) |
| **MILP master** | CBC (CPU simplex) | cuOpt GPU (PDLP + B&B) |

The question this benchmark answers: *Does replacing CBC with cuOpt GPU as the MILP sub-solver speed up Bonmin's B-OA on large problems?*

---

## Problem Definition

At each size (n, k):

```
min   Σ_{i=1..n} (x_i − 1)²  +  Σ_{j=1..k} (z_j − 1.3)²      [L2 objective]

s.t.  x_i² + 0.5·z_{f(i)}  ≤  1.6          [non-linear ball constraint]
      Σ x_i  +  Σ z_j        ≥  n + k − 1   [linear coupling]

      x_i ∈ [0, 3]   continuous
      z_j ∈ {0,1,2,3}  integer
```

Group assignment: `f(i) = (i·k) // n`  (0-indexed; k groups of n/k variables each)  
**Known global optimum:** `x_i = 1`, `z_j = 1`  →  `f* = k · (1−1.3)² = 0.09·k`

---

## Algorithm — B-OA with OA Cuts

```
Phase 0  NLP relaxation (all z continuous) → initial OA cuts

Repeat (up to MAX_ITER=6):
  A. MILP master (cuOpt GPU):
       min  η
       s.t. η ≥ Σ 2(x̄_i−1)·x_i + Σ 2(z̄_j−1.3)·z_j + const_t    [T obj cuts]
            2x̄_i·x_i + 0.5·z_{f(i)} ≤ 1.6 + x̄_i²                [T·n ball cuts]
            Σx_i + Σz_j ≥ n+k−1
       → LB = η*,  integer candidate ẑ

  B. NLP fix (IPOPT):
       Fix z = ẑ,  solve continuous NLP
       → UB = min(UB, f(x̄, ẑ));  add new OA cuts at (x̄, ẑ)

  If NLP infeasible: add OA cuts at x̂_i = √max(0, 1.6 − 0.5·ẑ_{f(i)})
                     (tightest valid cut from the infeasible z point)

  Converge when (UB − LB) / max(1, |UB|) < 1e-3
```

**Key difference from standard B-OA:** when IPOPT declares the NLP fix infeasible (coupling constraint can't be met at the ball-constrained x_i values), ball tangent cuts at the maximum-feasible x̂_i are added. This prevents the MILP from revisiting the infeasible z region.

---

## Column Glossary

| Column | Meaning |
|--------|---------|
| **size** | Human-readable label matching Bonmin/cuOpt benchmarks. |
| **n** | Continuous variables `x_i ∈ [0, 3]`. |
| **k** | Integer variables `z_j ∈ {0, 1, 2, 3}`. |
| **iters** | Total B-OA outer iterations (each = 1 MILP + 1 NLP fix). Phase 0 NLP relaxation not counted. |
| **nlp** | Total IPOPT time: NLP relaxation + all NLP fix calls. |
| **milp-build** | Python time to construct the cuOpt `Problem` object (add variables + constraints). Grows as O(T·n) — one ball cut per x_i per iteration. |
| **milp-solve** | cuOpt GPU wall-clock for all MILP master solves combined. |
| **total** | End-to-end wall-clock: nlp + milp-build + milp-solve. |
| **objective** | Best feasible objective found (L2 norm at optimal x,z). `---` = no feasible point found. |
| **f\*** | Known analytical optimum = `0.09·k`. |
| **status** | `optimal` = converged with gap ≤ 1e-3.  `milp_infeasible` = MILP sub-solver timed out or became infeasible before gap closed (note: optimal answer may still have been found in an earlier iteration). |

---

## Results

```
    size       n   k iters       nlp  milp-build  milp-solve      total    objective     f*   status
────────────────────────────────────────────────────────────────────────────────────────────────────
    tiny      10   1     2       33ms          0ms       1703ms      1736ms       0.0900 0.0900   optimal
   small      50   1     2       28ms          1ms       1527ms      1556ms       0.0900 0.0900   optimal
  small+     200   2     3       52ms          5ms       2294ms      2352ms       0.1800 0.1800   optimal
 medium-     600   2     3      109ms         16ms       2299ms      2425ms       0.1800 0.1800   optimal
  medium    1500   2     3      236ms         61ms       3490ms      3789ms       0.1800 0.1800   optimal
 medium+    3000   2     4      642ms        338ms      11866ms     12852ms       0.1800 0.1800   optimal
  large-    5000   4     4     1011ms        745ms      50966ms     52733ms       0.3600 0.3600   optimal
   large   10000   4     4     2545ms       2456ms     169584ms    174606ms       0.3600 0.3600   optimal
  large+   20000   4     4     5231ms       9238ms     388343ms    402858ms       0.3600 0.3600   milp_infeasible*
  xlarge   50000   6     1     6365ms      11691ms     301850ms    319922ms          --- 0.5400   milp_infeasible
────────────────────────────────────────────────────────────────────────────────────────────────────
```

*large+: the algorithm found the correct optimum (obj = 0.3600 = f*) in an earlier iteration but the final MILP timed out before certifying LB = UB.  

**Solved optimally (certified): 8/10**  
**Total-time ratio (slowest / fastest optimal): 112×** (large 174.6 s vs small 1.6 s)

---

## Time Breakdown Analysis

### Where does the time go?

| size | nlp % | milp-build % | milp-solve % |
|------|--------|--------------|--------------|
| tiny | 1.9% | 0.0% | 98.1% |
| small | 1.8% | 0.1% | 98.1% |
| small+ | 2.2% | 0.2% | 97.6% |
| medium- | 4.5% | 0.7% | 94.8% |
| medium | 6.2% | 1.6% | 92.2% |
| medium+ | 5.0% | 2.6% | 92.3% |
| large- | 1.9% | 1.4% | 96.7% |
| large | 1.5% | 1.4% | 97.2% |

The MILP master solve dominates at every size. GPU CUDA initialisation (≈700 ms overhead) makes this especially pronounced for small problems.

### Scaling of MILP master (cumulative ball cuts)

At convergence with T iterations, the MILP master for iteration T has:

| Size | n | T | T·n ball cuts | n·T cost vs n=10 |
|------|---|---|---------------|------------------|
| tiny | 10 | 2 | 20 | 1× |
| medium | 1500 | 3 | 4500 | 225× |
| medium+ | 3000 | 4 | 12000 | 600× |
| large- | 5000 | 4 | 20000 | 1000× |
| large | 10000 | 4 | 40000 | 2000× |

The T·n growth in constraints is the fundamental scalability limit of this naive B-OA implementation. Each iteration rebuilds the entire MILP from scratch with one new set of n ball cuts added.

---

## Comparison: Hybrid B-OA vs Bonmin vs Direct cuOpt MIP

| size | n | k | **Bonmin** | **Direct cuOpt MIP** | **Hybrid B-OA** |
|------|---|---|-----------|----------------------|-----------------|
| tiny | 10 | 1 | 13.7 ms | 1 058 ms | 1 736 ms |
| small | 50 | 1 | 9.6 ms | 693 ms | 1 556 ms |
| small+ | 200 | 2 | 83.1 ms | 702 ms | 2 352 ms |
| medium- | 600 | 2 | 124.2 ms | 709 ms | 2 425 ms |
| medium | 1500 | 2 | 492.7 ms | 816 ms | 3 789 ms |
| medium+ | 3000 | 2 | 1 043 ms | 2 256 ms | 12 852 ms |
| large- | 5000 | 4 | 33 908 ms | 2 276 ms | 52 733 ms |
| large | 10000 | 4 | *(> 30 s)* | 2 817 ms | 174 606 ms |
| large+ | 20000 | 4 | *(> 30 s)* | 6 402 ms | 402 858 ms* |
| xlarge | 50000 | 6 | *(> 30 s)* | 19 203 ms | 319 922 ms† |

*Found optimal answer but MILP timed out before certifying lower bound.  
†MILP timed out at the very first master solve (50 000 ball cuts from NLP relaxation).

### Winner at each size

| size | Winner | Why |
|------|--------|-----|
| n ≤ 3000 | **Bonmin** | No GPU overhead; CBC dual simplex fast for small LP relaxations |
| n = 5000–10000 | **Direct cuOpt MIP** | Single shot: no B-OA outer loop, 1 MILP solve with compact formulation |
| n ≥ 20000 | **Direct cuOpt MIP** | Hybrid's T·n cut explosion; direct MIP stays compact |

**The direct cuOpt MIP beats the hybrid at every tested size.** The hybrid is never fastest because:
1. GPU initialisation overhead (~700 ms) hurts small problems (Bonmin wins there).
2. Accumulating T·n ball cuts makes the MILP larger than the direct formulation at large n.
3. Multiple sequential GPU round-trips (one per iteration) can't amortise the initialisation cost.

---

## Why the Direct MIP Beats the Hybrid

The direct cuOpt MIP (from `mip_cuopt_scaling.py`) solves a **linearised version** of the problem in a single shot:

| Property | Direct cuOpt MIP | Hybrid B-OA (this benchmark) |
|---|---|---|
| MILP variables | 2n + 2k (aux vars for L1) | n + k + 1 (per MILP solve) |
| MILP constraints | 3n + 2k + 1 | T·(n+1) + 1 grows each iteration |
| Number of GPU round-trips | **1** | T (one per B-OA iteration) |
| Non-linear treatment | Dropped (exact linear reformulation) | OA cuts (approximation, converges) |
| Objective | L1 (approximation of L2) | L2 exact |

The direct MIP trades objective precision (L1 vs L2) for a compact one-shot solve. The hybrid preserves the exact L2 objective and non-linear constraint but pays the cost of T sequential GPU calls each with a growing MILP.

---

## Scalability Limit of This Implementation

The key scalability bottleneck is the **T·n ball cuts** accumulated in the MILP master:

```
After T iterations: T × n ball cuts, each involving 2 variables (x_i, z_{f(i)})
Python addConstraint loop: ~90–290 µs / constraint  →  O(T·n) Python overhead
PDLP solve time: grows with constraint count
```

### How to fix it

**1. Aggregate by group (exact, eliminates the O(n) scaling):**  
Since all x_i in group g are symmetric (same z_{f(i)}, same objective coefficient), at optimality they're all equal. Replace n x_i variables with k group representatives X_g. The MILP collapses from O(n) to O(k) variables and cuts — trivially fast at any n.

**2. Use a faster MILP sub-solver (parallel CPU):**  
HiGHS (`pip install highspy`) uses parallel revised dual simplex, warm-starts between iterations, and avoids GPU latency. For the small 2k+1 variable MILP from fix #1, any solver would be fast.

**3. Use SHOT (Supporting Hyperplane Optimization Toolkit):**  
SHOT is a dedicated MINLP-OA solver that implements exactly this algorithm with proper feasibility cuts, warm-starting, and efficient cut management. Available as a Pyomo sub-solver.

---

## Reproducibility

```bash
source .venv/bin/activate
python hybrid_boa_scaling.py
```

Dependencies:
- `cuopt-cu13==26.4.0` (GPU MILP solver)
- `pyomo` + IPOPT 3.13.2 (NLP solver, from IDAES)
- Runtime: CUDA 13.0, NVIDIA GB10

Algorithm settings used:
- B-OA convergence tolerance: 1e-3 (relative gap)
- Maximum outer iterations: 6
- MILP time limit per solve: 300 s
- IPOPT tolerance: 1e-9, max 3000 iterations
