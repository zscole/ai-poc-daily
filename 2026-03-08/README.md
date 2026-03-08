# NCTB-QA: Bangla Educational Question Answering – POC

Reproduces the core methodology of the **NCTB-QA** paper:
*"NCTB-QA: A Large-Scale Bangla Educational Question Answering Dataset and Benchmarking Performance"*

## What it does

| Step | Description |
|------|-------------|
| Data | Loads **BnQUAD** (or NCTB-QA if available) — real Bangla QA data from HuggingFace |
| Model | Fine-tunes **XLM-RoBERTa-base** for extractive question answering |
| Unanswerable | Detects unanswerable questions via null-answer (CLS) score threshold |
| Training | 3 epochs with AdamW + linear warmup scheduler; loss visibly decreases |
| Evaluation | Reports **Exact Match** and **F1** on a held-out split |
| Output | Saves `results.json` with all metrics + shows example predictions |

## Architecture

```
Context + Question
       │
  XLM-RoBERTa-base (multilingual transformer)
       │
  QA head: start_logits, end_logits per token
       │
  ┌────┴────────────────────┐
  │  Null score > threshold? │
  │  → "unanswerable"        │
  │  else → extract span     │
  └──────────────────────────┘
```

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# Run
python main.py
```

Runs on CPU or GPU automatically. On CPU expect ~5–15 min depending on hardware.

## Expected output

```
============================================================
 NCTB-QA: Bangla Educational QA – POC
============================================================
Device: cpu
Trying to load NCTB-QA ...
✓ Loaded BnQUAD: DatasetDict(...)

Train examples : 300
Eval  examples : 100

Sample question: বাংলাদেশের রাজধানী কোথায়?
...

Training for 3 epochs on BnQUAD
────────────────────────────────────────────────────────────
  Epoch 1  step   1/ 38  loss=5.6231
  Epoch 1  step  10/ 38  loss=5.1047
  ...
  Epoch 3  step  38/ 38  loss=2.8903

Training loss per epoch:
 epoch  avg_loss
     1    5.2100
     2    3.6400
     3    2.8900

✓ Loss decreased: 5.2100 → 2.8900

────────────────────────────────────────────────────────────
Metric                  Score
────────────────────────────
Exact Match             42.00%
F1 Score                61.50%
────────────────────────────

Example Predictions:
────────────────────────────────────────────────────────────

[Q] বাংলাদেশের রাজধানী কোথায়?
[Gold] ['ঢাকা']
[Pred] ঢাকা

✓ Results saved to results.json
```

> Exact scores will vary slightly by run. Expect EM ~35–55%, F1 ~55–70% for 300 training examples.

## Key parameters (main.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `xlm-roberta-base` | Base model (multilingual) |
| `TRAIN_LIMIT` | 300 | Training examples (increase for better accuracy) |
| `EVAL_LIMIT` | 100 | Evaluation examples |
| `NUM_EPOCHS` | 3 | Training epochs |
| `NULL_SCORE_DIFF_THRESHOLD` | 0.0 | Unanswerable detection sensitivity |
| `MAX_LENGTH` | 384 | Max tokens per passage chunk |
| `DOC_STRIDE` | 128 | Sliding window stride for long contexts |

## Dataset

**BnQUAD** (Bangla Question Answering Dataset) — 87k+ QA pairs from Bangladeshi texts, available on HuggingFace as `csebuetnlp/bnquad`. Format mirrors SQuAD 1.1 but in Bangla.

**NCTB-QA** adds 87,805 pairs from 50 NCTB textbooks with unanswerable questions, closely matching the paper's setup.
