"""
Automatic Generation of High-Performance RL Environments
=========================================================
Uses Claude API to translate a reference RL environment into a
vectorized high-performance batch implementation, with hierarchical
verification and iterative LLM-assisted repair.
"""

import os
import sys
import re
import time
import textwrap
import tempfile
import subprocess
import numpy as np
import anthropic

# ---------------------------------------------------------------------------
# Reference CartPole environment (the "spec" — clean, readable Python)
# ---------------------------------------------------------------------------
REFERENCE_ENV_CODE = textwrap.dedent("""\
    import numpy as np

    class CartPoleEnv:
        \"\"\"Reference CartPole environment (single-instance, pure Python).\"\"\"

        GRAVITY          = 9.8
        MASS_CART        = 1.0
        MASS_POLE        = 0.1
        TOTAL_MASS       = MASS_POLE + MASS_CART
        HALF_POLE_LENGTH = 0.5
        POLE_MASS_LEN    = MASS_POLE * HALF_POLE_LENGTH
        FORCE_MAG        = 10.0
        TAU              = 0.02
        THETA_THRESHOLD  = 12 * 2 * np.pi / 360
        X_THRESHOLD      = 2.4

        def __init__(self):
            self.state = None

        def reset(self, seed=None):
            rng = np.random.RandomState(seed)
            self.state = rng.uniform(low=-0.05, high=0.05, size=(4,))
            return self.state.copy()

        def step(self, action):
            x, x_dot, theta, theta_dot = self.state
            force = self.FORCE_MAG if action == 1 else -self.FORCE_MAG

            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            temp      = (force + self.POLE_MASS_LEN * theta_dot**2 * sin_t) / self.TOTAL_MASS
            theta_acc = (self.GRAVITY * sin_t - cos_t * temp) / (
                self.HALF_POLE_LENGTH * (4.0/3.0 - self.MASS_POLE * cos_t**2 / self.TOTAL_MASS)
            )
            x_acc = temp - self.POLE_MASS_LEN * theta_acc * cos_t / self.TOTAL_MASS

            x         = x         + self.TAU * x_dot
            x_dot     = x_dot     + self.TAU * x_acc
            theta     = theta     + self.TAU * theta_dot
            theta_dot = theta_dot + self.TAU * theta_acc

            self.state = np.array([x, x_dot, theta, theta_dot])

            terminated = bool(
                x < -self.X_THRESHOLD or x > self.X_THRESHOLD
                or theta < -self.THETA_THRESHOLD or theta > self.THETA_THRESHOLD
            )
            reward = 1.0 if not terminated else 0.0
            return self.state.copy(), reward, terminated, {}
""")

# ---------------------------------------------------------------------------
# Prompt template (the "generic recipe" from the paper)
# Sentinels __REF_CODE__ and __ERR_SECTION__ avoid str.format() issues.
# ---------------------------------------------------------------------------
GENERATION_PROMPT_TEMPLATE = textwrap.dedent("""\
    You are an expert in high-performance computing and reinforcement learning environments.

    Below is a REFERENCE implementation of a CartPole RL environment in Python:

    ```python
    __REF_CODE__
    ```

    ## Task
    Generate a VECTORIZED, high-performance `BatchCartPoleEnv` class that runs **N environments
    in parallel** using NumPy. Requirements:

    1. Constructor: `__init__(self, n_envs: int)`
    2. `reset(self, seed=None) -> np.ndarray`  shape (n_envs, 4) — seeds each env with seed+i
    3. `step(self, actions: np.ndarray) -> (states, rewards, dones, infos)`
       - actions shape: (n_envs,), values 0 or 1
       - states shape: (n_envs, 4), rewards/dones shape: (n_envs,)
    4. Use **identical physics constants and equations** as the reference.
    5. Use NumPy vectorization — no Python for-loops in `reset` or `step`.
    6. Store internal state in `self.states` (shape n_envs x 4).
    __ERR_SECTION__

    Return ONLY the Python class definition wrapped in ```python ... ```. No explanations.
""")

# ---------------------------------------------------------------------------
# Verification harness (runs in a subprocess for isolation)
# Sentinels __REF__ / __BATCH__ avoid str.format() brace-escaping issues.
# ---------------------------------------------------------------------------
VERIFICATION_SCRIPT = textwrap.dedent("""\
    import sys
    import numpy as np
    import time

    # ---- inject reference env ----
    __REF__

    # ---- inject generated batch env ----
    __BATCH__

    N_ENVS  = 16
    N_STEPS = 100
    SEED    = 7

    # ------------------------------------------------------------------
    # Tier 1: Shape / API contract
    # ------------------------------------------------------------------
    print("Tier 1: API contract ...", flush=True)
    benv = BatchCartPoleEnv(n_envs=N_ENVS)
    states = benv.reset(seed=SEED)
    assert states.shape == (N_ENVS, 4), f"reset shape {states.shape} != ({N_ENVS}, 4)"
    actions = np.zeros(N_ENVS, dtype=int)
    ns, rw, dn, _ = benv.step(actions)
    assert ns.shape == (N_ENVS, 4), f"step states shape {ns.shape}"
    assert rw.shape == (N_ENVS,),   f"rewards shape {rw.shape}"
    assert dn.shape == (N_ENVS,),   f"dones shape {dn.shape}"
    print("  PASS", flush=True)

    # ------------------------------------------------------------------
    # Tier 2: Numerical equivalence against N reference envs
    # ------------------------------------------------------------------
    print("Tier 2: Numerical equivalence ...", flush=True)

    ref_envs = [CartPoleEnv() for _ in range(N_ENVS)]
    ref_states = np.array([e.reset(seed=SEED + i) for i, e in enumerate(ref_envs)])

    benv2 = BatchCartPoleEnv(n_envs=N_ENVS)
    benv2.reset(seed=SEED)        # initialise internals
    benv2.states = ref_states.copy()  # force identical start

    rng = np.random.RandomState(SEED)
    for step_i in range(N_STEPS):
        acts = rng.randint(0, 2, size=N_ENVS)

        # Reference step
        ref_res  = [e.step(int(a)) for e, a in zip(ref_envs, acts)]
        ref_ns   = np.array([r[0] for r in ref_res])
        ref_rw   = np.array([r[1] for r in ref_res])
        ref_dn   = np.array([r[2] for r in ref_res])

        # Batch step
        b_ns, b_rw, b_dn, _ = benv2.step(acts)

        if not np.allclose(ref_ns, b_ns, atol=1e-6):
            max_err = np.max(np.abs(ref_ns - b_ns))
            print(f"  FAIL step {step_i}: states diverge, max_err={max_err:.2e}", flush=True)
            sys.exit(1)
        if not np.allclose(ref_rw, b_rw, atol=1e-6):
            print(f"  FAIL step {step_i}: rewards differ", flush=True)
            sys.exit(1)
        if not np.array_equal(ref_dn, b_dn):
            print(f"  FAIL step {step_i}: dones differ", flush=True)
            sys.exit(1)

        # Re-sync terminated reference envs so they don't diverge next step
        for i, done in enumerate(ref_dn):
            if done:
                ref_envs[i] = CartPoleEnv()
                ref_envs[i].state = b_ns[i].copy()

    print(f"  PASS  ({N_STEPS} steps, {N_ENVS} envs, max state error < 1e-6)", flush=True)

    # ------------------------------------------------------------------
    # Tier 3: Performance benchmark
    # ------------------------------------------------------------------
    print("Tier 3: Performance benchmark ...", flush=True)

    BENCH_ENVS  = 1024
    BENCH_STEPS = 500

    # --- sequential reference ---
    seq_envs = [CartPoleEnv() for _ in range(BENCH_ENVS)]
    [e.reset(seed=i) for i, e in enumerate(seq_envs)]
    rng2 = np.random.RandomState(0)
    t0 = time.perf_counter()
    for _ in range(BENCH_STEPS):
        acts = rng2.randint(0, 2, size=BENCH_ENVS)
        for e, a in zip(seq_envs, acts):
            e.step(int(a))
    ref_time = time.perf_counter() - t0

    # --- vectorized batch ---
    bperf = BatchCartPoleEnv(n_envs=BENCH_ENVS)
    bperf.reset(seed=0)
    rng3 = np.random.RandomState(0)
    t0 = time.perf_counter()
    for _ in range(BENCH_STEPS):
        acts = rng3.randint(0, 2, size=BENCH_ENVS)
        bperf.step(acts)
    batch_time = time.perf_counter() - t0

    speedup = ref_time / batch_time
    print(f"  Sequential : {ref_time:.3f}s  ({BENCH_ENVS} envs x {BENCH_STEPS} steps)", flush=True)
    print(f"  Vectorized : {batch_time:.3f}s  ({BENCH_ENVS} envs x {BENCH_STEPS} steps)", flush=True)
    print(f"  Speedup    : {speedup:.1f}x", flush=True)

    if speedup < 1.0:
        print("  WARNING: batch implementation is slower than sequential", flush=True)
    else:
        print(f"  PASS  ({speedup:.1f}x faster than sequential Python)", flush=True)

    print("\\nALL TIERS PASSED", flush=True)
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_code_block(text: str) -> str:
    """Pull the first ```python ... ``` block from an LLM response."""
    m = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    return m.group(1) if m else text.strip()


def run_verification(reference_code: str, batch_code: str) -> tuple[bool, str, str]:
    """Execute the verification harness in a fresh subprocess."""
    # Use sentinel replacement (not str.format) so Python code with {braces}
    # in the injected source doesn't cause KeyError/IndexError.
    script = (
        VERIFICATION_SCRIPT
        .replace("__REF__", reference_code)
        .replace("__BATCH__", batch_code)
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as fh:
        fh.write(script)
        path = fh.name
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=120,
        )
        return result.returncode == 0, result.stdout, result.stderr
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  AUTO-GENERATION OF HIGH-PERFORMANCE RL ENVIRONMENTS")
    print("=" * 65)
    print()

    client = anthropic.Anthropic()

    # ------------------------------------------------------------------
    # Phase 1: Describe reference environment
    # ------------------------------------------------------------------
    print("Phase 1 — Reference environment")
    print("  Environment : CartPole (4D state, 2 actions, Euler integration)")
    print("  Spec        : pure-Python, single-instance, ~40 lines")
    print()

    # ------------------------------------------------------------------
    # Phase 2: LLM-based generation with iterative repair
    # ------------------------------------------------------------------
    MAX_ATTEMPTS  = 4
    batch_code    = None
    verified      = False
    error_section = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"Phase 2 — LLM generation  (attempt {attempt}/{MAX_ATTEMPTS})")

        prompt = (
            GENERATION_PROMPT_TEMPLATE
            .replace("__REF_CODE__", REFERENCE_ENV_CODE)
            .replace("__ERR_SECTION__", error_section)
        )

        t0 = time.perf_counter()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = time.perf_counter() - t0

        raw_text   = response.content[0].text
        batch_code = extract_code_block(raw_text)
        n_lines    = batch_code.count("\n") + 1
        in_tok     = response.usage.input_tokens
        out_tok    = response.usage.output_tokens

        print(f"  Generated   : {n_lines} lines")
        print(f"  Tokens      : {in_tok} in / {out_tok} out  ({elapsed:.1f}s)")
        print()

        # --------------------------------------------------------------
        # Phase 3: Hierarchical verification
        # --------------------------------------------------------------
        print(f"Phase 3 — Verification  (attempt {attempt})")
        ok, stdout, stderr = run_verification(REFERENCE_ENV_CODE, batch_code)

        # Print subprocess output with indentation
        for line in stdout.splitlines():
            print(f"  {line}")
        if stderr.strip():
            for line in stderr.strip().splitlines()[:15]:
                print(f"  ERR: {line}")

        if ok:
            print()
            print("  >>> ALL TIERS PASSED <<<")
            verified = True
            break

        # --------------------------------------------------------------
        # Phase 4: Iterative repair — feed errors back to LLM
        # --------------------------------------------------------------
        print()
        print(f"Phase 4 — Repair feedback to LLM")
        combined_err = (stderr + "\n" + stdout)[-1500:]
        error_section = textwrap.dedent(f"""\
            ## Previous attempt failed — please fix:
            ```
            {combined_err}
            ```
        """)
        print(f"  Feeding {len(combined_err)} chars of error back to LLM ...")
        print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Status        : {'SUCCESS ✓' if verified else 'PARTIAL (max attempts reached)'}")
    print(f"  Attempts used : {attempt}")
    print(f"  Model         : claude-haiku-4-5-20251001")
    print()

    if batch_code:
        print("Generated BatchCartPoleEnv (first 40 lines):")
        print("-" * 65)
        for line in batch_code.splitlines()[:40]:
            print(line)
        remaining = batch_code.count("\n") + 1 - 40
        if remaining > 0:
            print(f"  ... ({remaining} more lines)")

    print()
    print("=" * 65)
    print("  WORKFLOW DEMONSTRATED")
    print("=" * 65)
    print("  1. Reference spec  → LLM prompt (generic template)")
    print("  2. LLM output      → BatchCartPoleEnv class")
    print("  3. Tier-1 verify   → API contract / shapes")
    print("  4. Tier-2 verify   → Numerical equivalence vs reference")
    print("  5. Tier-3 verify   → Performance benchmark (speedup)")
    print("  6. On failure      → Error fed back to LLM for repair")
    print()


if __name__ == "__main__":
    main()
