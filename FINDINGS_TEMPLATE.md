# Findings: LB4OMP auto-selection on a zero-imbalance FEM kernel

> Fill in after running the experiment on x86. This template structures
> the writeup — replace the placeholders with real numbers.

## Setup

- **Hardware**: [CPU model, cores, clock speed, cache sizes]
- **Software**: [clang version, LB4OMP commit, Linux kernel]
- **Kernel**: 3D sum-factorized mass operator, degree [P], [N] elements,
  [T] time steps, [C] threads
- **FLOPs/element**: [X], total/step: [Y]

## Result 1: Static is optimal (confirming the hypothesis)

| Configuration        | Steady wall (μs) | Overhead vs static | Imbalance |
|----------------------|------------------:|-------------------:|----------:|
| baseline_static      |              [X]  |              0.0%  |     [X]%  |
| lb4omp_static        |              [X]  |           [+X.X]%  |     [X]%  |
| lb4omp_dynamic1      |              [X]  |           [+X.X]%  |     [X]%  |
| lb4omp_guided        |              [X]  |           [+X.X]%  |     [X]%  |
| lb4omp_rl_q_learning |              [X]  |           [+X.X]%  |     [X]%  |

## Result 2: RL convergence behavior

- Q-Learning converged at step [N] / did not converge.
- After convergence, per-thread element counts were [equal/unequal],
  indicating the selected schedule was [static/DLS-name].
- Steady-state overhead after convergence: [X]% above static.
- During exploration (steps 0–[K]): wall time was [X]–[Y]× the static
  baseline, caused by [schedule names tried].

## Result 3: LB4OMP's static path overhead

- Running LB4OMP with `OMP_SCHEDULE=static` added [X]% overhead versus
  the system libomp with static.
- This measures LB4OMP's instrumentation/bookkeeping cost when no
  auto-selection is active.

## Key finding

[One paragraph: does the auto-selection cheaply recognize the
static-optimal case? What is the cost of not recognizing it? Is the
residual overhead after convergence significant for a real PDE
time-stepping code (hundreds to thousands of operator applies)?]

## Reproducibility

Full source, build scripts, and raw CSV data at:
[link to repo or archive]
