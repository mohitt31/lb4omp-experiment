# Findings: LB4OMP auto-selection on a zero-imbalance FEM kernel

## Setup

- **Hardware**: Intel Xeon @ 2.20 GHz, 2 cores, Google Cloud Shell
- **Software**: clang (system), LB4OMP master branch, Linux (Debian-based)
- **Kernel**: 3D sum-factorized mass operator, degree 4, 20000 elements,
  200 time steps, 2 threads
- **FLOPs/element**: 76875 (12 * 5^4 + 5^3), total/step: 1.5375 GFLOP

## Result 1: Static is optimal (confirming the hypothesis)

| Configuration        | Steady wall (us) | Overhead vs static | Imbalance |
|----------------------|-----------------:|-------------------:|----------:|
| baseline_static      |           86830  |              0.0%  |     2.3%  |
| lb4omp_static        |           86668  |             -0.2%  |     1.9%  |
| lb4omp_dynamic1      |          104950  |            +20.9%  |     0.0%  |
| lb4omp_guided        |           86148  |             -0.8%  |     0.0%  |
| lb4omp_rl_q_learning |           87936  |             +1.3%  |     2.5%  |
| lb4omp_rl_double_q   |           87536  |             +0.8%  |     1.9%  |
| lb4omp_rl_sarsa      |           87247  |             +0.5%  |     2.1%  |

Static and guided are effectively tied at the top. Dynamic with chunk
size 1 pays a 20.9% penalty — pure scheduling overhead on identical
work. All three RL auto-selection policies converge within 0.5-1.3% of
static, confirming that the selector does eventually find a low-overhead
schedule.

## Result 2: RL convergence behavior

- Q-Learning showed stable performance (~87 ms/step) for the first 134
  steps, then entered an **exploration episode at steps 135-144** where
  wall time spiked to **3.08 million us** (35x the static baseline).
  Recovery was gradual: step 145 returned to normal (~93 ms).
- A second, smaller exploration spike occurred at steps 176-179
  (up to 156 ms, ~1.8x baseline).
- After convergence, per-thread element counts remained equal
  (10000/10000), consistent with a static or near-static schedule.
- Steady-state overhead after convergence: +1.3% above static.
- Double-Q and SARSA showed similar patterns with smaller exploration
  spikes and lower residual overhead (+0.8% and +0.5% respectively).

## Result 3: LB4OMP's static path overhead

- Running LB4OMP with `OMP_SCHEDULE=static` added **-0.2%** overhead
  versus the system libomp with static — effectively zero.
- LB4OMP's instrumentation and bookkeeping cost is negligible when no
  auto-selection is active. The runtimes are within noise.

## Key finding

The RL-based auto-selection does eventually converge to a near-static
schedule on this zero-imbalance kernel, with residual overhead of only
0.5-1.3%. However, the **exploration cost is severe**: Q-Learning's
worst exploration episode (steps 135-144) consumed ~25 million us of
wall time that pure static would have completed in ~870K us — a 28x
penalty across those 10 steps. In a real PDE time-stepping code running
hundreds of operator applies, this exploration phase would represent a
non-trivial one-time cost. The key question for practical use is whether
the exploration budget (epsilon, decay rate) can be tuned to avoid
testing obviously-bad schedules (like dynamic with tiny chunks) on
kernels where the per-iteration timing variance is already near zero —
i.e., whether the selector can use low variance as a signal that the
current schedule is already near-optimal.

SARSA showed the best behavior: lowest residual overhead (+0.5%) and
less aggressive exploration, suggesting it may be more suitable than
Q-Learning for kernels where the cost of exploration is high relative
to any potential scheduling gain.

## Reproducibility

Full source, build scripts, and raw CSV data at:
https://github.com/mohitt31/lb4omp-experiment
