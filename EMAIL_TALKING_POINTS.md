# Email talking points for Mohit

> This is NOT a paste-ready email. Rewrite in your own words.
> Audience: Prof. Ciorba + CC Reto Krummenacher.
> Tone: respectful, concrete, no overselling.

## Subject line idea
Something like: "Sum-factorized FEM kernel results with LB4OMP"

## Key points to cover

1. **What you did**: Built a standalone experiment testing LB4OMP's
   auto-selection on a purely static/uniform kernel — 3D
   sum-factorized mass operator where every element does identical
   work (zero load imbalance by construction).

2. **Setup**: 20000 elements, degree 4, 2 threads on Intel Xeon
   @ 2.20 GHz, 200 time steps. Compared system libomp static vs
   LB4OMP static vs dynamic,1 vs guided vs RL (Q-Learning,
   Double-Q, SARSA).

3. **What you found** (use these numbers):
   - LB4OMP's static path adds zero overhead vs system libomp (-0.2%,
     within noise). Their instrumentation is lightweight.
   - RL auto-selection converges near static — residual overhead is
     only +0.5% (SARSA) to +1.3% (Q-Learning). Good result for LB4OMP.
   - BUT: Q-Learning hit a 35x exploration spike (steps 135-144,
     wall time jumped from 87ms to 3 million us) before recovering.
     This is ~25M us of wasted compute in 10 steps.
   - SARSA had the smoothest convergence and lowest overhead.
   - dynamic,1 confirmed as worst case: +20.9% pure overhead.

4. **Frame it positively**: The auto-selection DOES work on this kernel —
   it converges to the right answer. The interesting question is the
   exploration cost: can epsilon/decay be tuned so the selector
   recognizes "low timing variance = already near-optimal" and skips
   aggressive exploration?

5. **Offer**: Share the reproducible setup (repo link). Offer to:
   - Run on more cores (2 cores is limited — behavior may differ at
     8-16 cores where scheduling choices matter more)
   - Try different RL hyperparameters (epsilon, decay, reward metric)
   - Try different kernel sizes (vary degree or element count)
   - Integrate into their benchmark suite if useful

6. **Question to keep thread alive**: Ask if they'd like to run it on
   their own hardware (multi-socket, many-core) to see how the
   convergence behavior scales. Ask if they have a preferred RL
   configuration they'd recommend for low-imbalance workloads.

## What NOT to say
- Don't say their tool is broken — it works! Frame exploration cost as
  an optimization opportunity, not a flaw.
- Don't overstate: 1.3% overhead after convergence is small. The story
  is about the exploration phase, not the steady state.
- Don't mention internship/position in this email — pure technical.
- Don't mention AI/Claude/LLM anywhere.

## Repo to share
https://github.com/mohitt31/lb4omp-experiment
