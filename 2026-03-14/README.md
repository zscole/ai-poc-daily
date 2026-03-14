# Automatic Generation of High-Performance RL Environments

A working POC of the recipe described in the paper: use an LLM with a
generic prompt template, hierarchical verification, and iterative
agent-assisted repair to automatically produce semantically equivalent,
high-performance RL environments.

## What it does

1. **Reference spec** — a clean, readable single-instance CartPole environment
   (~40 lines of pure Python) serves as the ground-truth specification.

2. **LLM generation** — Claude Haiku receives a generic prompt template
   instructing it to produce a vectorized `BatchCartPoleEnv` that runs N
   environments in parallel using NumPy (no Python loops in hot path).

3. **Hierarchical verification** (3 tiers, run in a subprocess):
   - **Tier 1 – API contract**: shapes and dtypes match the expected interface.
   - **Tier 2 – Numerical equivalence**: 16 batch envs stepped 100 times must
     match 16 independent reference envs to within 1e-6.
   - **Tier 3 – Performance**: 1 024 envs × 500 steps benchmarked; speedup
     over sequential Python is reported.

4. **Iterative repair** — if any tier fails, the error output is fed back to
   the LLM (up to 4 attempts) until all tiers pass.

## How to run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
python main.py
```

Requires `ANTHROPIC_API_KEY` to be set. Completes in under 3 minutes on CPU.

## Expected output

```
=================================================================
  AUTO-GENERATION OF HIGH-PERFORMANCE RL ENVIRONMENTS
=================================================================

Phase 1 — Reference environment
  Environment : CartPole (4D state, 2 actions, Euler integration)
  Spec        : pure-Python, single-instance, ~40 lines

Phase 2 — LLM generation  (attempt 1/4)
  Generated   : 52 lines
  Tokens      : 612 in / 487 out  (1.4s)

Phase 3 — Verification  (attempt 1)
  Tier 1: API contract ...
    PASS
  Tier 2: Numerical equivalence ...
    PASS  (100 steps, 16 envs, max state error < 1e-6)
  Tier 3: Performance benchmark ...
    Sequential : 1.843s  (1024 envs x 500 steps)
    Vectorized : 0.092s  (1024 envs x 500 steps)
    Speedup    : 20.0x
    PASS  (20.0x faster than sequential Python)

  ALL TIERS PASSED

  >>> ALL TIERS PASSED <<<

=================================================================
  SUMMARY
=================================================================
  Status        : SUCCESS ✓
  Attempts used : 1
  Model         : claude-haiku-4-5-20251001
```

Typical speedup is **10–30×** on CPU for 1 024 parallel environments.

## Cost

Each generation attempt uses ~600 input tokens and ~500 output tokens with
Claude Haiku. At current pricing this is well under **$0.01 per run**.

## Files

| File | Description |
|------|-------------|
| `main.py` | Full pipeline: reference env, prompt, generation, verification, repair |
| `requirements.txt` | `anthropic`, `numpy` |
| `README.md` | This file |
