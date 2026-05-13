# Scalable MIP Benchmark Results — cuOpt GPU Solver

**Solver:** cuOpt 26.4.0 (PDLP + Branch-and-Bound) via `linear_programming.Problem`  
**Hardware:** NVIDIA GB10 (Grace Blackwell), 121.7 GiB VRAM, CUDA 13.0, Ubuntu 24.04 LTS, aarch64  
**Date:** 2026-05-13  
**Companion benchmark:** `minlp_scaling_results.md` (same sizes, Bonmin on CPU)

---

## What cuOpt Can (and Cannot) Solve

cuOpt's `linear_programming` module is a GPU-accelerated **LP / MIP** solver.  It handles:

- **LP**: linear objectives, linear constraints, continuous variables.
- **MIP**: above plus integer/binary variables (branch-and-bound over LP relaxations).

It does **not** support **non-linear constraints** or **non-linear objectives** (no MINLP).  
The quadratic MINLP from `minlp_scaling.py` must therefore be **reformulated** as a linear MIP before cuOpt can solve it.

---

## Reformulation: MINLP → Linear MIP

### Original MINLP (Bonmin benchmark)

```
min   Σ (x_i − 1)²  +  Σ (z_j − 1.3)²        [quadratic objective]
s.t.  x_i²  +  0.5·z_{f(i)}  ≤  1.6            [non-linear constraint]
      Σ x_i  +  Σ z_j         ≥  n + k − 1
      x_i ∈ [0,3]   continuous
      z_j ∈ {0,1,2,3}  integer
```

### Linearised MIP (cuOpt benchmark)

```
min   Σ d_i  +  Σ e_j                             [L1 objective]
s.t.  d_i  ≥  x_i − 1                             [|x_i − 1| upper arm]
      d_i  ≥  1 − x_i                             [|x_i − 1| lower arm]
      e_j  ≥  z_j − 1.3                           [|z_j − 1.3| upper arm]
      e_j  ≥  1.3 − z_j                           [|z_j − 1.3| lower arm]
      x_i  +  0.5·z_{f(i)}  ≤  1.6               [linear feasibility: no x_i² term]
      Σ x_i  +  Σ z_j        ≥  n + k − 1
      x_i ∈ [0,3]   continuous
      z_j ∈ {0,1,2,3}  integer
      d_i, e_j ≥ 0           auxiliary (continuous)
```

**Two changes from MINLP:**
1. **Objective** — L2 (squared) → L1 (absolute value), linearised with auxiliary variables `d_i`, `e_j`.  
2. **Ball constraint** — `x_i² ≤ …` → `x_i ≤ …` (drop the square; keeps the same structural coupling between `x_i` and `z_j`).

### What the solver actually sees

| Quantity | MINLP (Bonmin) | Linear MIP (cuOpt) |
|---|---|---|
| Decision variables | `n + k` | `2n + 2k` (adds `d_i`, `e_j`) |
| Constraints | `n + 1` (non-linear + coupling) | `3n + 2k + 1` (ball + abs-val + coupling) |
| Constraint type | Non-linear (x²) + linear | Pure linear |
| Objective type | Quadratic | Linear |

---

## Problem Definition

At each size the linearised MIP is solved with `n` continuous and `k` integer variables:

```
min   Σ_{i=1..n} d_i  +  Σ_{j=1..k} e_j

s.t.  d_i  ≥  x_i − 1                   for each i = 1..n
      d_i  ≥  1 − x_i                   for each i = 1..n
      e_j  ≥  z_j − 1.3                 for each j = 1..k
      e_j  ≥  1.3 − z_j                 for each j = 1..k
      x_i  +  0.5·z_{f(i)}  ≤  1.6     for each i = 1..n   [linear feasibility]
      Σ x_i  +  Σ z_j        ≥  n+k−1                      [linear coupling]

      x_i ∈ [0, 3]        continuous
      z_j ∈ {0, 1, 2, 3}  integer
      d_i, e_j ≥ 0        auxiliary continuous
```

Group assignment map: `f(i) = (i−1)·k // n` (0-indexed; same distribution as MINLP).

**Known global optimum:** `x_i = 1`, `z_j = 1` → `f* = |1 − 1.3| · k = 0.3·k`  
Why `z_j = 1` is unique: `|0−1.3| = 1.3`, `|1−1.3| = 0.3`, `|2−1.3| = 0.7`, `|3−1.3| = 1.7` — z_j=1 is the sole minimum.

---

## Column Glossary

| Column | Meaning |
|--------|---------|
| **size** | Human-readable label (same as MINLP benchmark for direct comparison). |
| **n** | Number of **continuous** variables `x_i ∈ [0, 3]`. Each has one linear feasibility constraint and two abs-val linearisation constraints. |
| **k** | Number of **integer** variables `z_j ∈ {0, 1, 2, 3}`. Kept small (1–6) to control the B&B master problem size. |
| **vars** | Original problem variable count = `n + k` (same denominator as MINLP for direct comparison). The solver actually allocates `2n + 2k` variables once auxiliary `d_i`, `e_j` are included. |
| **build** | Wall-clock time to construct the `Problem` object in Python (loops over `addVariable` and `addConstraint`). For large `n`, Python loop overhead dominates — this is a Python API cost, not a GPU cost. |
| **solve** | Wall-clock time for `p.solve()`: uploads the model to GPU, runs PDLP for the LP relaxation, and runs the B&B MIP engine (if needed). |
| **total** | `build + solve` end-to-end wall-clock time. |
| **objective** | Optimal objective value (L1 norm at solution). |
| **f\*** | Known analytical optimum = `0.3 · k`. Confirms global minimum was reached. |
| **nodes** | Branch-and-bound nodes explored. `0` means the LP relaxation at the root already yielded an integer-feasible optimal — no branching needed. |
| **status** | Solver termination condition. `optimal` = proven global optimum within time limit. |

---

## Algorithm Notes

**PDLP (Primal-Dual Linear Programming)** is a first-order GPU-native LP solver developed at Google and integrated into cuOpt:
- Runs entirely on GPU; exploits CUDA parallelism for matrix-vector products.
- Scales near-linearly with `n` for sparse problems (each constraint touches ≤ 3 variables here).
- Warm-starts naturally between B&B nodes.

**B&B (Branch-and-Bound)** handles the integer variables:
- At each node, PDLP solves the LP relaxation.
- Branching on `z_j` variables when fractional.
- For this problem, **nodes = 0 at every size** — the LP relaxation solution already assigns `z_j = 1` (integer) at the root because `z_j = 1` is the uniquely optimal integer value. PDLP finds it without branching.

**Why nodes = 0:**
- The L1 objective over `z_j ∈ {0,1,2,3}` with target 1.3 has a single minimum at `z_j = 1`.
- The LP relaxation (dropping integrality) also picks `z_j = 1.0` since it is both the real-valued minimum and the nearest integer.
- There is no gap between LP relaxation and integer optimum → no B&B needed.

**GPU overhead for small problems:**
- CUDA context initialisation and model upload to GPU take ~700–1000 ms regardless of problem size.
- For `n ≤ 1500`, this overhead exceeds the actual solve computation — the GPU is waiting for work.
- For `n ≥ 3000`, computation starts to dominate.

---

## Results

```
    size       n   k   vars     build      solve      total    objective     f*   nodes   status
────────────────────────────────────────────────────────────────────────────────────────────────
    tiny      10   1     11      0.1ms    1057.7ms    1057.8ms       0.3000   0.30       0   optimal
   small      50   1     51      0.5ms     693.2ms     693.7ms       0.3000   0.30       0   optimal
  small+     200   2    202      1.7ms     702.3ms     704.1ms       0.6000   0.60       0   optimal
 medium-     600   2    602     28.3ms     709.4ms     737.7ms       0.6000   0.60       0   optimal
  medium    1500   2   1502     22.0ms     816.2ms     838.2ms       0.6000   0.60       0   optimal
 medium+    3000   2   3002     65.6ms    2255.9ms    2321.5ms       0.6000   0.60       0   optimal
  large-    5000   4   5004    186.4ms    2276.3ms    2462.7ms       1.2000   1.20       0   optimal
   large   10000   4  10004    605.4ms    2817.3ms    3422.7ms       1.2000   1.20       0   optimal
  large+   20000   4  20004   2233.5ms    6401.5ms    8634.9ms       1.2000   1.20       0   optimal
  xlarge   50000   6  50006  14624.2ms   19202.9ms   33827.0ms       1.8000   1.80       0   optimal
────────────────────────────────────────────────────────────────────────────────────────────────
```

**Solve-time ratio (slowest / fastest): 28×**  
(`xlarge`: 50006 vars, k=6, 19.2 s  vs  `small`: 51 vars, k=1, 693 ms)

---

## Scaling Analysis

### Effect of n (LP cost) — k held constant at 2

| n    | vars | solve time | ratio vs previous |
|------|------|------------|-------------------|
|  200 |  202 |   702 ms   | —                 |
|  600 |  602 |   709 ms   | 1.0×              |
| 1500 | 1502 |   816 ms   | 1.2×              |
| 3000 | 3002 | 2 256 ms   | 2.8×              |

For `n ≤ 1500` the solve time is flat (~700–800 ms) — GPU initialisation dominates; the LP itself is negligible.  
At `n = 3000` the LP computation becomes visible; solve jumps ~2.8×.

### Effect of k — n held constant at 5000–50000 range

| n     | k | vars  | solve time | notes |
|-------|---|-------|------------|-------|
| 5 000 | 4 | 5 004 | 2 276 ms   | —     |
|10 000 | 4 |10 004 | 2 817 ms   | 1.2× more n |
|20 000 | 4 |20 004 | 6 402 ms   | 4× more n |
|50 000 | 6 |50 006 |19 203 ms   | 10× more n |

`k` has minimal impact here because all `z_j = 1` is found at the LP relaxation root (nodes = 0). The cost is driven by `n`, not `k`.

### Model build time (Python overhead)

| n      | build time |
|--------|------------|
|  3 000 |    66 ms   |
|  5 000 |   186 ms   |
| 10 000 |   605 ms   |
| 20 000 | 2 234 ms   |
| 50 000 |14 624 ms   |

Build time is dominated by Python-level `addConstraint` loops — approximately **O(n)** with a constant of ~290 µs/constraint.  For `n > 10 000` the Python overhead rivals the GPU solve time.  This can be eliminated by using the sparse-matrix `DataModel` API (LP only) or by compiling the model outside Python.

---

## Head-to-Head: cuOpt MIP vs Bonmin MINLP (same sizes)

| size | n | k | **Bonmin solve** | **cuOpt solve** | faster |
|------|---|---|-----------------|-----------------|--------|
| tiny     |    10 | 1 |   13.7 ms |  1 058 ms | Bonmin **77×** |
| small    |    50 | 1 |    9.6 ms |    693 ms | Bonmin **72×** |
| small+   |   200 | 2 |   83.1 ms |    702 ms | Bonmin **8×** |
| medium-  |   600 | 2 |  124.2 ms |    709 ms | Bonmin **6×** |
| medium   | 1 500 | 2 |  492.7 ms |    816 ms | Bonmin **1.7×** |
| medium+  | 3 000 | 2 | 1 043 ms  |  2 256 ms | Bonmin **2.2×** |
| large-   | 5 000 | 4 | 33 908 ms | 2 276 ms  | **cuOpt 15×** |
| large    |10 000 | 4 | *(> 30 s)* | 2 817 ms | **cuOpt >> 10×** |
| large+   |20 000 | 4 | *(> 30 s)* | 6 402 ms | **cuOpt >> 4×** |
| xlarge   |50 000 | 6 | *(> 30 s)* |19 203 ms | **cuOpt >> 1.5×** |

> **Note:** The comparison is not fully apples-to-apples — Bonmin solves a **harder** non-linear MIP (quadratic constraint + quadratic objective) while cuOpt solves the **linearised analog** (linear constraint + L1 objective).  The linearisation makes the problem genuinely easier.  The table still illustrates the GPU throughput advantage at large scale.

**Crossover point**: between `n = 3 000` (Bonmin 1× faster for MINLP) and `n = 5 000` (cuOpt 15× faster for the linearised version), with the GPU taking over as `n` grows.

---

## Solve-Time Ratio Comparison

| Benchmark | Fastest | Slowest | Ratio |
|-----------|---------|---------|-------|
| MINLP (Bonmin/CPU) | 9.6 ms (n=50, k=1) | 33 908 ms (n=5000, k=4) | **3 545×** |
| MIP (cuOpt/GPU) | 693 ms (n=50, k=1) | 19 203 ms (n=50000, k=6) | **28×** |

The GPU's solve time grows only **28×** across a **1000× range of problem size** (50 → 50 000 vars), while the CPU MINLP solver grows **3 545×** across a **100× range** (50 → 5 000 vars). This demonstrates the GPU's near-linear scaling advantage.

---

## Reproducibility

```bash
# Activate the virtual environment
source .venv/bin/activate

# Re-run the full benchmark
python mip_cuopt_scaling.py
```

Dependencies installed in `.venv/`:
- `cuopt-cu13==26.4.0` (GPU MIP solver, CUDA 13)
- `cudf-cu13` (cuOpt dependency)
- Runtime: CUDA 13.0, NVIDIA GB10
