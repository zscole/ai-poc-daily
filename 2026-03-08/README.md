# Judge Reliability Harness

Stress-tests the reliability of LLM judges across four independent dimensions using real API calls — no mocks, no hardcoded scores.

## What it does

Given a benchmark of question/response pairs at known quality levels, the harness asks a judge model (claude-haiku) to evaluate them and measures:

| Test | What it measures |
|------|-----------------|
| **Self-Consistency** | Same input judged N times — do scores agree? |
| **Position Bias** | Pairwise comparison with order swapped — does position affect preference? |
| **Calibration** | Known quality ordering (excellent > good > poor > wrong) — Spearman rank correlation |
| **Perturbation Robustness** | Minor prompt-prefix rewording — score stability |

Each dimension produces a 0–1 score, combined into a **composite reliability score**.

## Benchmark

Three task formats are included:
- **Free-response** — explain photosynthesis
- **Coding** — write a prime-checking function
- **Agentic** — evaluate a multi-step agent execution trace

Each item has four response variants: `excellent`, `good`, `poor`, `wrong`.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

## Run

```bash
python main.py
```

Completes in ~1–2 minutes on a standard connection (~90 API calls to claude-haiku-4-5).

## Expected output

```
╔══════════════════════════════════════════════════════════╗
║       Judge Reliability Harness — LLM Judge Stress Test  ║
╚══════════════════════════════════════════════════════════╝

TEST 1/4: Self-Consistency  (3 trials per item)
  [01/12] photosynthesis       excellent  trial 1: ordinal=5  binary=good
  ...

TEST 2/4: Position Bias
  photosynthesis        excellent  vs poor       │ fwd=A  rev=B  consistent=True
  ...

TEST 3/4: Calibration
  photosynthesis        excellent  → score=5  (expected rank 4)
  ...

TEST 4/4: Perturbation Robustness
  photosynthesis        good   prefix 1: score=4
  ...

══════════════════════════════════════════════════════════════
  JUDGE RELIABILITY HARNESS — FINAL REPORT
══════════════════════════════════════════════════════════════
  Self-Consistency    (agree across trials)         0.917  [PASS]
  Position Fairness   (pairwise order swap)         1.000  [PASS]
  Calibration         (Spearman rank corr.)         0.875  [PASS]
  Perturbation Robust (prompt variation)            0.875  [PASS]
──────────────────────────────────────────────────────────────
  ── COMPOSITE RELIABILITY ──                       0.917  [PASS]
══════════════════════════════════════════════════════════════
```

Detailed results are written to `results.json`.

## Scoring

- **PASS** ≥ 0.70 — judge is reliable on this dimension
- **WARN** 0.50–0.69 — moderate reliability issues
- **FAIL** < 0.50 — significant reliability problems

## Extending the harness

- Add items to `BENCHMARK` in `main.py` (any number of formats/qualities)
- Change `MODEL` to swap in a different judge
- Increase `SELF_CONSISTENCY_TRIALS` for stronger consistency estimates
- Add prompt prefixes to `PROMPT_PREFIXES` for broader perturbation coverage
