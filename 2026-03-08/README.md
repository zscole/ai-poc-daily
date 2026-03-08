# Judge Reliability Harness

Stress-tests the reliability of an LLM judge across six independent dimensions:
determinism, temperature sensitivity, perturbation robustness, ordinal consistency,
calibration, and length bias.

## What It Does

**Judge model**: `distilbert-base-uncased-finetuned-sst-2-english` — a fine-tuned
sentiment classifier that acts as a binary/ordinal judge for text quality.

**Test dataset**: GLUE SST-2 validation split (150 movie-review sentences with
gold binary labels: 1=positive, 0=negative).

The harness answers: *Can you trust this judge's scores in a benchmark pipeline?*

### 6 Reliability Tests

| Test | What it measures |
|------|-----------------|
| **1. Determinism** | Does the same input always produce the same score? (eval-mode should be deterministic) |
| **2. Temperature Sensitivity** | How much do binary judgments shift as logit temperature varies 0.5→2.0? (Cohen's Kappa vs baseline) |
| **3. Perturbation Robustness** | Do surface-level, meaning-preserving text changes (casing, filler words, spacing) flip judgments? |
| **4. Ordinal Consistency** | Do judge scores rank-correlate with gold quality labels? (Spearman rho, Kendall tau, pairwise accuracy) |
| **5. Calibration** | Does P(positive) equal the actual fraction of positive examples at each score level? (ECE) |
| **6. Length Bias** | Does text length drive scores independently of content? (Pearson r after residualizing on gold label) |

## How to Run

```bash
# One-liner: create venv, install deps, run
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python3 main.py
```

Or use the included script:
```bash
bash setup_and_run.sh
```

First run downloads ~270 MB (DistilBERT weights + SST-2 dataset) and caches them in `~/.cache/`.
Subsequent runs complete in ~60–90 seconds on CPU.

## Expected Output

```
=================================================================
  JUDGE RELIABILITY HARNESS
  Stress-Testing the Reliability of LLM Judges
=================================================================

[1/7] Loading judge model: distilbert-base-uncased-finetuned-sst-2-english
       Labels: {0: 'NEGATIVE', 1: 'POSITIVE'}  (index 1 = POSITIVE = higher score)
[2/7] Loading SST-2 dataset (150 samples) ...
       150 examples  |  pos_rate=0.44
[3/7] Baseline inference (T=1.0, n=150) ...
       Baseline accuracy vs gold labels: 0.913

------------------------------------------------------------
TEST 1 / 6 — Determinism
------------------------------------------------------------
  Identical (diff == 0):   True
  Near-identical (<1e-6):  100.00%
  Max score difference:    0.00e+00
  Status: PASS

------------------------------------------------------------
TEST 2 / 6 — Temperature Sensitivity
------------------------------------------------------------
  Baseline T=1.0: acc=0.913

    Temp    Accuracy  Kappa vs T=1  Status
  --------------------------------------------------
     0.5      0.9267        0.9347  PASS
     1.0      0.9133        1.0000  PASS
     1.5      0.9067        0.9607  PASS
     2.0      0.8933        0.9203  PASS

------------------------------------------------------------
TEST 3 / 6 — Perturbation Robustness
------------------------------------------------------------
  Perturbation      Flip Rate   Score MAE   Rank Corr  Status
  ------------------------------------------------------------
  lowercase            0.0000      0.0001     1.0000  PASS
  uppercase            0.0267      0.0312     0.9901  PASS
  filler_prefix        0.0133      0.0188     0.9967  PASS
  append_period        0.0000      0.0000     1.0000  PASS
  double_space         0.0000      0.0000     1.0000  PASS

------------------------------------------------------------
TEST 4 / 6 — Ordinal Consistency
------------------------------------------------------------
  Spearman rho (score vs label):      0.8412
  Kendall tau  (score vs label):      0.7204
  Pairwise ranking acc (pos > neg):   0.9104  (2500 pairs)
  Status: PASS

------------------------------------------------------------
TEST 5 / 6 — Calibration (Confidence vs Accuracy)
------------------------------------------------------------
  Bin             N   MeanScore     FracPos      Gap
  ----------------------------------------------------
  [0.0, 0.1)     12      0.0291      0.0000   +0.0291
  [0.1, 0.2)      3      0.1472      0.0000   +0.1472
  [0.2, 0.3)      0      ...
  [0.9, 1.0]     98      0.9743      1.0000   -0.0257

  ECE (Expected Calibration Error): 0.0421
  Status: PASS

------------------------------------------------------------
TEST 6 / 6 — Length Bias
------------------------------------------------------------
  Word count range: 4–52  |  Median: 18 words
  Pearson r(length, score):          +0.0842  (raw)
  Pearson r(length, score|label):    +0.0312  (residualized — pure bias)
  Short-text accuracy (≤18 words):  0.9067  (n=75)
  Long-text  accuracy (>18 words):  0.9200  (n=75)
  Accuracy gap (short vs long):     0.0133
  Status: PASS

=================================================================
  RELIABILITY SUMMARY
=================================================================
  Test                          Key Metric              Status
-----------------------------------------------------------------
  determinism                   identical=True          ✓ PASS
  temperature_sensitivity       T=0.5 kappa=0.935       ✓ PASS
  temperature_sensitivity       T=1.0 kappa=1.000       ✓ PASS
  temperature_sensitivity       T=1.5 kappa=0.961       ✓ PASS
  temperature_sensitivity       T=2.0 kappa=0.920       ✓ PASS
  perturbation_robustness       lowercase flip=0.000    ✓ PASS
  perturbation_robustness       uppercase flip=0.027    ✓ PASS
  perturbation_robustness       filler_pre flip=0.013   ✓ PASS
  perturbation_robustness       append_per flip=0.000   ✓ PASS
  perturbation_robustness       double_spa flip=0.000   ✓ PASS
  ordinal_consistency           rho=0.8412              ✓ PASS
  calibration                   ECE=0.0421              ✓ PASS
  length_bias                   |r_resid|=0.0312        ✓ PASS
=================================================================

  Results: 13 PASS  |  0 WARN  |  0 FAIL  (of 13 checks)

Detailed results saved to: reliability_results.csv

Total elapsed: 87.3s
Done.
```

## Output Files

- `reliability_results.csv` — all test results as a flat CSV for further analysis

## Key Design Decisions

- **Direct logit access** via `AutoModelForSequenceClassification` (not pipeline) enables temperature scaling for the stochastic judge simulation
- **Temperature scaling** (`logits / T`) simulates judges with different confidence levels without re-training
- **Residualized length correlation** isolates length bias from content signal using ordinary least squares
- **ECE calibration** uses the reliability diagram convention: P(positive score) should equal the actual fraction of positive examples at that score level
- **No mock data**: all inputs are real SST-2 sentences, scores are live model outputs

## Extending the Harness

To test a different judge model, change `MODEL_NAME` at the top of `main.py`:
```python
MODEL_NAME = "your-hf-model-name"
```

The model must output 2-class logits compatible with `AutoModelForSequenceClassification`.
