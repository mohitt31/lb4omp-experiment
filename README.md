# LB4OMP Scheduling Experiment: Zero-Imbalance FEM Kernel

Does LB4OMP's automatic scheduling algorithm selection recognize a
perfectly balanced workload cheaply, or does it pay measurable overhead
versus plain `schedule(static)`?

## Background

Sum-factorized tensor-product FEM kernels parallelize over elements. On a
Cartesian mesh with uniform polynomial degree, **every element performs
identical work** — load imbalance is zero by construction. The optimal
schedule is `static` with equal-sized chunks: one assignment, no atomics,
no profiling. Any dynamic loop scheduling (DLS) technique adds overhead
that cannot be recouped because there is no imbalance to fix.

LB4OMP (U Basel, Ciorba group) extends LLVM's OpenMP runtime with 13+
DLS techniques and RL-based automatic selection. The auto-selection
profiles loop execution times over several invocations and converges to a
schedule. **Open question (stated by Prof. Ciorba): they have not tested
their comparative selection on a purely static/uniform kernel.**

This experiment fills that gap.

## What the harness measures

`sumfact_harness.cpp` implements a standalone 3D sum-factorized
mass-operator apply (evaluate → scale → integrate), parallelized over
elements with `#pragma omp parallel for schedule(runtime)`.

Per time step it records:
- **Wall time** (microseconds)
- **Per-thread active time** (compute + scheduling overhead, no barrier wait)
- **Per-thread element count** (reveals the scheduling pattern)
- **Load imbalance** = (max − min) / mean thread time × 100%
- **GFLOP/s** (model: 12·m⁴ + m³ FLOPs per element, m = degree + 1)

## How to run

### 1. Build the harness

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

On a system where the default compiler lacks OpenMP (e.g., macOS with
AppleClang), point CMake at a compiler that has it:

```bash
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release
```

### 2. Build LB4OMP

```bash
./scripts/setup_lb4omp.sh
```

Clones and builds `unibas-dmi-hpc/LB4OMP` into `./lb4omp/`. Requires
clang.

### 3. Run the experiment

```bash
CPU_MHZ=2500 ./scripts/run_experiment.sh
```

Set `CPU_MHZ` to your CPU's clock speed (`grep MHz /proc/cpuinfo`).
Tune with environment variables:

| Variable    | Default | What                        |
|-------------|--------:|-----------------------------|
| `DEGREE`    |       4 | Polynomial degree           |
| `ELEMENTS`  |   20000 | Number of mesh elements     |
| `TIMESTEPS` |     200 | Time steps (loop iterations)|
| `CPU_MHZ`   |    2500 | CPU clock (required by LB4OMP)|

Runs these configurations and writes CSV per-step data to `results/`:

| Config              | OMP_SCHEDULE | Runtime     | What it tests              |
|---------------------|-------------|-------------|----------------------------|
| baseline_static     | static      | system libomp | Optimal baseline           |
| lb4omp_static       | static      | LB4OMP      | LB4OMP's static path overhead |
| lb4omp_dynamic1     | dynamic,1   | LB4OMP      | Worst-case DLS overhead    |
| lb4omp_guided       | guided      | LB4OMP      | Standard guided overhead   |
| lb4omp_rl_q_learning| 6           | LB4OMP      | RL auto-selection (Q-Learning) |
| lb4omp_rl_double_q  | 7           | LB4OMP      | RL auto-selection (Double-Q)   |
| lb4omp_rl_sarsa     | 8           | LB4OMP      | RL auto-selection (SARSA)      |

### 4. View results

```bash
python3 scripts/summarize.py
```

Prints a comparison table with steady-state wall time, overhead vs
baseline, and load imbalance.

## Target platform

Run on **x86-64 Linux** (EPYC / Xeon). LB4OMP targets x86 LLVM OpenMP.
GitHub Codespaces with 4+ cores work well.

**Do not run on macOS ARM** — LB4OMP is not tested there, and the
scheduling overhead characteristics differ.

## Kernel correctness

The kernel is a standard sum-factorization:
- `apply_1d(S, U, V)`: 1D tensor contraction, V = S·U
- `element_apply()`: three sequential `apply_1d` calls (evaluate),
  quadrature scaling, three `apply_1d_T` calls (integrate)
- All elements use the same shape matrix S and the same m³ data size

Zero imbalance follows from identical per-element work — no
data-dependent branching, no variable iteration counts.

## Interpreting results

**What to look for in the RL auto-selection runs:**

1. **Convergence**: do the per-step wall times start high (exploration)
   and drop toward the static baseline? If so, at which step?
2. **Converged schedule**: do the per-thread element counts become equal
   (= static) after convergence, or stay variable (= DLS)?
3. **Residual overhead**: is the steady-state wall time equal to static,
   or higher? The gap is the irreducible profiling/selection cost.

**Expected outcome (hypothesis):**

On this kernel, `static` is optimal. A good auto-selector should:
- Converge to static (or an equivalent equal-chunk schedule) quickly
- Add minimal residual overhead after convergence

If the selector does NOT converge to static, that's equally interesting
— it means the profiling signal (which shows zero imbalance) is not
sufficient for the selector to recognize the static-optimal case.

## References

- LB4OMP repo: https://github.com/unibas-dmi-hpc/LB4OMP
- Korndörfer et al., "LB4OMP: A Dynamic Load Balancing Library for
  Multithreaded Applications," IEEE TPDS 33(4), 2021.
- Mohammed et al., "Automated Scheduling Algorithm Selection and Chunk
  Parameter Calculation in OpenMP," IEEE TPDS 33(12), 2022.
