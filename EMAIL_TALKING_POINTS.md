# Email draft — talking points for Mohit

> This is NOT a paste-ready email. Rewrite in your own words.
> Audience: Prof. Ciorba + CC Reto Krummenacher.
> Tone: respectful, concrete, no overselling.

## Subject line idea
Something like: "Sum-factorized FEM kernel results with LB4OMP"

## Key points to cover

1. **What you did**: Ran the experiment she suggested — tested LB4OMP's
   auto-selection on a purely static/uniform kernel (3D sum-factorized
   mass operator, all elements identical work).

2. **Setup in one sentence**: standalone C++ harness, [N] elements,
   degree [P], [C] threads on [hardware], compared schedule(static)
   baseline vs LB4OMP auto-selection (Q-Learning, SARSA).

3. **What you found** (fill in with real numbers):
   - Did the RL converge to static? At what step?
   - What was the overhead during exploration?
   - What was the residual overhead after convergence?
   - Did LB4OMP's static path itself add overhead vs system libomp?

4. **Offer**: Share the reproducible setup (source + scripts + raw
   data). Offer to run additional configurations if they want (different
   RL policies, different reward metrics, different kernel sizes).

5. **Question to keep the thread alive**: Ask if they have a preferred
   way to characterize this — e.g., would they want the harness
   integrated into their own benchmark suite, or is a standalone
   reproduction enough? Is there a specific RL configuration they'd
   recommend trying?

## What NOT to say
- Don't claim their tool is broken — frame it as "untested case" (their
  own words), and you're providing the test.
- Don't overstate the finding — if the overhead is 2%, say 2%, don't
  call it "significant" unless it actually is for a real PDE solve.
- Don't mention internship/position aspirations in this email — keep it
  purely technical. The relationship builds by doing good work.
