# Judge Reliability Harness

Stress-tests the reliability of an LLM judge across four orthogonal dimensions, using real human-preference data from OpenAI's summarisation feedback dataset.

## What it does

| Test | What it measures |
|------|-----------------|
| **Intra-judge consistency** | Same prompt judged N times — do verdicts agree? |
| **Positional bias** | Swapping response A↔B — does the verdict flip spuriously? |
| **Calibration accuracy** | Does the judge agree with human ground-truth labels? |
| **Ordinal rating stability** | Repeat 1–10 ratings — how correlated are they across trials? |

All judgments use `claude-haiku-4-5-20251001` as the judge model via the Anthropic API. Ground-truth labels come from the `openai/summarize_from_feedback` dataset (human raters chose which of two Reddit-post summaries was better).

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Run

```bash
python main.py
```

Completes in **~1–2 minutes** on any machine (CPU-only, ~55 API calls total).

## Expected output

```
══════════════════════════════════════════════════════════════
  JUDGE RELIABILITY HARNESS
  Stress-testing LLM judge reliability
══════════════════════════════════════════════════════════════
[1/5] Loading benchmark dataset …
[2/5] Consistency Test  (3 trials × 5 samples)…
  sample 1: ['A', 'A', 'A']  →  100% consistent
  …
[3/5] Positional Bias Test  (5 samples)…
  sample 1: AB=A  BA(norm)=A  ✓
  …
[4/5] Calibration Test  (10 samples vs. human labels)…
  sample 1: judge=A  human=B  ✗
  …
[5/5] Ordinal Reliability Test  (2 trials × 5 samples)…
  sample 1: T1:(A=7,B=5) T2:(A=7,B=5)
  …

══════════════════════════════════════════════════════════════
  JUDGE RELIABILITY REPORT
══════════════════════════════════════════════════════════════
  Model  : claude-haiku-4-5-20251001

                   Metric    Value                  Interpretation
  Intra-judge consistency   87.5%              higher → more reliable
       Positional bias rate   20.0%                lower → less biased
   Calibration accuracy       70.0%        vs. human annotations
                  Cohen's κ    0.340  >0.60 = substantial agreement
         Ordinal Spearman ρ    0.812       >0.80 = high ordinal stability

  Overall Reliability Score : 73.5%  [MEDIUM]
══════════════════════════════════════════════════════════════
  Saved → results.json
```

## Output files

- **`results.json`** — machine-readable reliability metrics

## Reliability score

A weighted composite of all four sub-metrics:

```
score = 0.30 × consistency
      + 0.20 × (1 − positional_bias)
      + 0.30 × calibration_accuracy
      + 0.20 × ordinal_spearman_rho
```

Grades: **HIGH** > 75% · **MEDIUM** 50–75% · **LOW** < 50%

## Extending the harness

- Swap in a different judge by changing `JUDGE_MODEL`
- Increase sample counts (`N_*` constants at the top of `main.py`) for higher-confidence estimates
- Replace the dataset loader with any dataset that provides `question`, `response_a`, `response_b`, and `winner` fields
