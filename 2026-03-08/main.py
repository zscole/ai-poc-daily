"""
Judge Reliability Harness
=========================
Stress-tests an LLM judge (DistilBERT fine-tuned on SST-2) across six
reliability dimensions: determinism, temperature sensitivity, perturbation
robustness, ordinal consistency, calibration, and length bias.

Reference: "Judge Reliability Harness: Stress Testing the Reliability of
LLM Judges" — validates that automated judge scores are trustworthy before
using them in benchmarks or RLHF pipelines.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
import glob as _glob

def _bootstrap_deps():
    """Add sibling POC venv to sys.path if numpy isn't available."""
    try:
        import numpy  # noqa: F401
    except ImportError:
        staging = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for sp in sorted(_glob.glob(os.path.join(staging, "*/.venv/lib/python3*/site-packages"))):
            if sp not in sys.path:
                sys.path.insert(0, sp)

_bootstrap_deps()

import itertools
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.metrics import cohen_kappa_score, accuracy_score

# ── Configuration ─────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased-finetuned-sst-2-english"
N_SAMPLES    = 150
BATCH_SIZE   = 32
MAX_LENGTH   = 128
TEMPERATURES = [0.5, 1.0, 1.5, 2.0]
RESULTS_CSV  = "reliability_results.csv"


# ── Helpers ───────────────────────────────────────────────────

def _stat(r):
    """Extract statistic value from a scipy result (works across versions)."""
    return float(getattr(r, "statistic", r[0]))


def _pval(r):
    """Extract p-value from a scipy result."""
    return float(getattr(r, "pvalue", r[1]))


# ── Model & Data Loading ───────────────────────────────────────

def load_judge_model(model_name: str):
    """Load tokenizer and classification model in eval mode on CPU."""
    print(f"[1/7] Loading judge model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    id2label = model.config.id2label
    print(f"       Labels: {id2label}  (index 1 = POSITIVE = higher judge score)")
    return tokenizer, model


def load_dataset_samples(n: int = N_SAMPLES):
    """
    Load SST-2 validation split. Tries standalone sst2 first,
    falls back to glue/sst2. Returns (texts, labels).
    """
    print(f"[2/7] Loading SST-2 dataset ({n} samples) ...")
    try:
        ds = load_dataset("sst2", split=f"validation[:{n}]")
    except Exception:
        ds = load_dataset("glue", "sst2", split=f"validation[:{n}]")
    texts  = [ex["sentence"] for ex in ds]
    labels = np.array([ex["label"] for ex in ds], dtype=int)
    pos_rate = labels.mean()
    print(f"       {len(texts)} examples  |  pos_rate={pos_rate:.2f}")
    return texts, labels


# ── Core Inference Engine ─────────────────────────────────────

def batch_infer(
    tokenizer,
    model,
    texts: list,
    temperature: float = 1.0,
) -> tuple:
    """
    Run batched inference with temperature scaling applied to logits
    before softmax. Returns (scores, preds) as numpy arrays.

    Temperature scaling: scaled_logits = logits / T
      T < 1.0 → sharper (more confident)
      T > 1.0 → softer (more uncertain)
      T = 1.0 → unmodified
    """
    all_scores = []
    n = len(texts)

    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
            )
            logits = model(**enc).logits          # (batch, 2) raw logits
            scaled = logits / temperature          # temperature scaling
            probs  = F.softmax(scaled, dim=-1)     # (batch, 2) probabilities
            # Index 1 = POSITIVE class score
            scores = probs[:, 1].cpu().numpy()
            all_scores.append(scores)

    scores = np.concatenate(all_scores)            # (N,)
    preds  = (scores >= 0.5).astype(int)
    return scores, preds


# ── Reliability Tests ─────────────────────────────────────────

def test_determinism(tokenizer, model, texts: list) -> dict:
    """
    Test 1: Determinism
    Run identical inputs twice. With dropout disabled (eval mode + no_grad),
    the model must produce byte-identical scores every time.
    Failure here indicates non-deterministic execution or unexpected stochasticity.
    """
    print("\n" + "-" * 60)
    print("TEST 1 / 6 — Determinism")
    print("-" * 60)

    scores1, _ = batch_infer(tokenizer, model, texts)
    scores2, _ = batch_infer(tokenizer, model, texts)

    diff         = np.abs(scores1 - scores2)
    max_diff     = float(diff.max())
    identical    = bool(np.all(diff == 0.0))
    near_ident   = float(np.mean(diff < 1e-6))

    status = "PASS" if identical else ("WARN" if max_diff < 1e-5 else "FAIL")
    print(f"  Identical (diff == 0):   {identical}")
    print(f"  Near-identical (<1e-6):  {near_ident:.2%}")
    print(f"  Max score difference:    {max_diff:.2e}")
    print(f"  Status: {status}")

    return {
        "test": "determinism",
        "identical": identical,
        "near_identical_rate": near_ident,
        "max_score_diff": max_diff,
        "status": status,
    }


def test_temperature_sensitivity(
    tokenizer, model, texts: list, labels: np.ndarray
) -> list:
    """
    Test 2: Temperature Sensitivity
    Scale logits by T in {0.5, 1.0, 1.5, 2.0}. Measure agreement with T=1.0
    baseline (Cohen's Kappa) and accuracy vs. gold labels.
    Models heavily dependent on calibration (e.g. for ordinal grading) should
    show high kappa across temperature ranges.
    """
    print("\n" + "-" * 60)
    print("TEST 2 / 6 — Temperature Sensitivity")
    print("-" * 60)

    # Baseline at T=1.0
    base_scores, base_preds = batch_infer(tokenizer, model, texts, temperature=1.0)
    base_acc = accuracy_score(labels, base_preds)
    print(f"  Baseline T=1.0: acc={base_acc:.3f}\n")
    print(f"  {'Temp':>6}  {'Accuracy':>10}  {'Kappa vs T=1':>14}  {'Status'}")
    print(f"  {'-'*50}")

    results = []
    for T in TEMPERATURES:
        scores, preds = batch_infer(tokenizer, model, texts, temperature=T)
        acc   = accuracy_score(labels, preds)
        kappa = cohen_kappa_score(base_preds.tolist(), preds.tolist())
        status = "PASS" if kappa > 0.8 else ("WARN" if kappa > 0.5 else "FAIL")
        print(f"  {T:>6.1f}  {acc:>10.4f}  {kappa:>14.4f}  {status}")
        results.append({
            "test": "temperature_sensitivity",
            "temperature": T,
            "accuracy": float(acc),
            "kappa_vs_T1": float(kappa),
            "status": status,
        })
    return results


# Semantic-preserving perturbations that shouldn't change quality judgments
PERTURBATIONS = {
    "lowercase":      lambda t: t.lower(),
    "uppercase":      lambda t: t.upper(),
    "filler_prefix":  lambda t: "Well, " + t,
    "append_period":  lambda t: t.rstrip(".") + ".",
    "double_space":   lambda t: t.replace(" ", "  "),
}


def test_perturbation_robustness(
    tokenizer, model, texts: list, labels: np.ndarray
) -> list:
    """
    Test 3: Perturbation Robustness
    Apply 5 semantic-preserving text transformations. The judge should be
    insensitive to surface-level noise that doesn't change meaning.
    Metrics: flip_rate (binary judgment changes), score MAE, rank correlation.
    """
    print("\n" + "-" * 60)
    print("TEST 3 / 6 — Perturbation Robustness")
    print("-" * 60)

    orig_scores, orig_preds = batch_infer(tokenizer, model, texts)

    print(f"  {'Perturbation':<16}  {'Flip Rate':>10}  {'Score MAE':>10}  {'Rank Corr':>10}  {'Status'}")
    print(f"  {'-'*60}")

    results = []
    for name, fn in PERTURBATIONS.items():
        perturbed = [fn(t) for t in texts]
        pert_scores, pert_preds = batch_infer(tokenizer, model, perturbed)

        flip_rate = float((orig_preds != pert_preds).mean())
        score_mae = float(np.abs(orig_scores - pert_scores).mean())
        rho       = _stat(spearmanr(orig_scores, pert_scores))

        status = "PASS" if flip_rate < 0.05 else ("WARN" if flip_rate < 0.15 else "FAIL")
        print(f"  {name:<16}  {flip_rate:>10.4f}  {score_mae:>10.4f}  {rho:>10.4f}  {status}")

        results.append({
            "test": "perturbation_robustness",
            "perturbation": name,
            "flip_rate": flip_rate,
            "score_mae": score_mae,
            "rank_corr": float(rho),
            "status": status,
        })
    return results


def test_ordinal_consistency(scores: np.ndarray, labels: np.ndarray) -> dict:
    """
    Test 4: Ordinal Consistency
    For a judge to be reliable in ordinal grading, it must assign higher
    scores to higher-quality responses. We measure:
    - Spearman rank correlation of judge scores vs gold labels
    - Kendall tau rank correlation
    - Pairwise ranking accuracy: across (pos, neg) pairs, how often does
      the judge rank the positive example higher?
    """
    print("\n" + "-" * 60)
    print("TEST 4 / 6 — Ordinal Consistency")
    print("-" * 60)

    sp_r  = _stat(spearmanr(scores, labels))
    kt_r  = _stat(kendalltau(scores, labels))

    # Pairwise ranking accuracy (positive should score > negative)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    # Cap at 50x50 = 2500 pairs for speed
    pairs = list(itertools.product(pos_idx[:50], neg_idx[:50]))
    correct = sum(1 for i, j in pairs if scores[i] > scores[j])
    pairwise_acc = correct / len(pairs) if pairs else 0.0

    status = "PASS" if sp_r > 0.7 else ("WARN" if sp_r > 0.4 else "FAIL")
    print(f"  Spearman rho (score vs label):      {sp_r:.4f}")
    print(f"  Kendall tau  (score vs label):      {kt_r:.4f}")
    print(f"  Pairwise ranking acc (pos > neg):   {pairwise_acc:.4f}  ({len(pairs)} pairs)")
    print(f"  Status: {status}")

    return {
        "test": "ordinal_consistency",
        "spearman_rho": float(sp_r),
        "kendall_tau": float(kt_r),
        "pairwise_ranking_acc": float(pairwise_acc),
        "status": status,
    }


def test_calibration(
    scores: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> dict:
    """
    Test 5: Calibration
    A well-calibrated judge's confidence (score) should equal the probability
    it is correct. We compute Expected Calibration Error (ECE) across
    10 equal-width bins of the score range.
    ECE = sum_b (|b|/N) * |mean_score_b - accuracy_b|
    """
    print("\n" + "-" * 60)
    print("TEST 5 / 6 — Calibration (Confidence vs Accuracy)")
    print("-" * 60)
    print(f"  {'Bin':>12}  {'N':>5}  {'MeanScore':>10}  {'FracPos':>10}  {'Gap':>8}")
    print(f"  {'-'*52}")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    bins_data = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (scores >= lo) & (scores < hi)
        # Include hi endpoint in last bin
        if hi == 1.0:
            mask = (scores >= lo) & (scores <= hi)

        n_bin = int(mask.sum())
        if n_bin == 0:
            bins_data.append({"lo": float(lo), "hi": float(hi), "n": 0,
                               "mean_score": None, "frac_pos": None, "gap": None})
            continue

        mean_score = float(scores[mask].mean())
        # Calibration: does P(positive) = actual fraction positive in bin?
        frac_pos   = float(labels[mask].mean())
        gap        = abs(mean_score - frac_pos)
        ece       += (n_bin / len(scores)) * gap

        bins_data.append({"lo": float(lo), "hi": float(hi), "n": n_bin,
                          "mean_score": mean_score, "frac_pos": frac_pos, "gap": gap})
        print(f"  [{lo:.1f}, {hi:.1f})  {n_bin:>5}  {mean_score:>10.4f}  {frac_pos:>10.4f}  {gap:>+8.4f}")

    status = "PASS" if ece < 0.05 else ("WARN" if ece < 0.1 else "FAIL")
    print(f"\n  ECE (Expected Calibration Error): {ece:.4f}")
    print(f"  Status: {status}")

    return {
        "test": "calibration",
        "ece": float(ece),
        "n_bins_populated": sum(1 for b in bins_data if b["n"] > 0),
        "status": status,
    }


def test_length_bias(
    scores: np.ndarray, labels: np.ndarray, texts: list
) -> dict:
    """
    Test 6: Length Bias
    A reliable judge should assess quality, not text length.
    We measure whether text length drives scores independently of content by:
    1. Computing raw Pearson r(word_count, score)
    2. Residualizing scores on labels (remove content signal), then re-measuring
    3. Comparing accuracy on short vs long texts (median split)
    """
    print("\n" + "-" * 60)
    print("TEST 6 / 6 — Length Bias")
    print("-" * 60)

    word_counts = np.array([len(t.split()) for t in texts], dtype=float)

    # Raw length-score correlation
    r_raw = _stat(pearsonr(word_counts, scores))

    # Residualize scores on gold label to isolate pure length effect
    X = np.column_stack([labels.astype(float), np.ones(len(labels))])
    coeffs, _, _, _ = np.linalg.lstsq(X, scores, rcond=None)
    residuals = scores - X @ coeffs
    r_resid = _stat(pearsonr(word_counts, residuals))

    # Short vs long text accuracy (median word-count split)
    median_wc  = np.median(word_counts)
    short_mask = word_counts <= median_wc
    long_mask  = ~short_mask
    preds      = (scores >= 0.5).astype(int)

    short_acc = float(accuracy_score(labels[short_mask], preds[short_mask])) if short_mask.sum() > 0 else float("nan")
    long_acc  = float(accuracy_score(labels[long_mask],  preds[long_mask]))  if long_mask.sum()  > 0 else float("nan")
    acc_gap   = abs(short_acc - long_acc)

    status = "PASS" if abs(r_resid) < 0.1 else ("WARN" if abs(r_resid) < 0.2 else "FAIL")

    print(f"  Word count range: {int(word_counts.min())}–{int(word_counts.max())}  "
          f"|  Median: {int(median_wc)} words")
    print(f"  Pearson r(length, score):          {r_raw:+.4f}  (raw)")
    print(f"  Pearson r(length, score|label):    {r_resid:+.4f}  (residualized — pure bias)")
    print(f"  Short-text accuracy (≤{int(median_wc)} words):  {short_acc:.4f}  (n={short_mask.sum()})")
    print(f"  Long-text  accuracy (>{int(median_wc)} words):  {long_acc:.4f}  (n={long_mask.sum()})")
    print(f"  Accuracy gap (short vs long):      {acc_gap:.4f}")
    print(f"  Status: {status}")

    return {
        "test": "length_bias",
        "r_length_score_raw": float(r_raw),
        "r_length_score_residual": float(r_resid),
        "short_acc": short_acc,
        "long_acc": long_acc,
        "acc_gap": float(acc_gap),
        "status": status,
    }


# ── Summary & Output ──────────────────────────────────────────

def flatten_results(all_results: list) -> list:
    """Flatten lists-of-dicts and single dicts into a flat list of rows."""
    rows = []
    for r in all_results:
        if isinstance(r, list):
            rows.extend(r)
        else:
            rows.append(r)
    return rows


def print_summary_table(rows: list):
    """Print a compact reliability summary table."""
    print("\n" + "=" * 65)
    print("  RELIABILITY SUMMARY")
    print("=" * 65)
    print(f"  {'Test':<28}  {'Key Metric':<22}  Status")
    print("-" * 65)

    STATUS_SYMBOL = {"PASS": "✓", "WARN": "~", "FAIL": "✗"}

    for row in rows:
        test   = row.get("test", "?")
        status = row.get("status", "?")
        sym    = STATUS_SYMBOL.get(status, "?")

        if test == "determinism":
            metric = f"identical={row.get('identical')}"
        elif test == "temperature_sensitivity":
            T = row.get("temperature", "?")
            k = row.get("kappa_vs_T1", 0)
            metric = f"T={T:.1f} kappa={k:.3f}"
        elif test == "perturbation_robustness":
            p = row.get("perturbation", "?")[:12]
            f = row.get("flip_rate", 0)
            metric = f"{p} flip={f:.3f}"
        elif test == "ordinal_consistency":
            metric = f"rho={row.get('spearman_rho', 0):.4f}"
        elif test == "calibration":
            metric = f"ECE={row.get('ece', 0):.4f}"
        elif test == "length_bias":
            metric = f"|r_resid|={abs(row.get('r_length_score_residual', 0)):.4f}"
        else:
            metric = ""

        print(f"  {test:<28}  {metric:<22}  {sym} {status}")

    print("=" * 65)

    # Compute overall pass rate
    statuses = [r.get("status") for r in rows]
    n_pass = statuses.count("PASS")
    n_warn = statuses.count("WARN")
    n_fail = statuses.count("FAIL")
    print(f"\n  Results: {n_pass} PASS  |  {n_warn} WARN  |  {n_fail} FAIL  "
          f"(of {len(statuses)} checks)")


def save_results(rows: list, path: str = RESULTS_CSV):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"\nDetailed results saved to: {path}")


# ── Main ──────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print("=" * 65)
    print("  JUDGE RELIABILITY HARNESS")
    print("  Stress-Testing the Reliability of LLM Judges")
    print("=" * 65)
    print()

    # ── Setup ──────────────────────────────────────────────────
    tokenizer, model = load_judge_model(MODEL_NAME)
    texts, labels    = load_dataset_samples(N_SAMPLES)

    # ── Baseline inference ─────────────────────────────────────
    print(f"[3/7] Baseline inference (T=1.0, n={N_SAMPLES}) ...")
    base_scores, base_preds = batch_infer(tokenizer, model, texts, temperature=1.0)
    base_acc = accuracy_score(labels, base_preds)
    print(f"       Baseline accuracy vs gold labels: {base_acc:.3f}")

    # ── Run all six reliability tests ─────────────────────────
    all_results = []

    r1 = test_determinism(tokenizer, model, texts)
    all_results.append(r1)

    r2 = test_temperature_sensitivity(tokenizer, model, texts, labels)
    all_results.append(r2)

    r3 = test_perturbation_robustness(tokenizer, model, texts, labels)
    all_results.append(r3)

    r4 = test_ordinal_consistency(base_scores, labels)
    all_results.append(r4)

    r5 = test_calibration(base_scores, labels)
    all_results.append(r5)

    r6 = test_length_bias(base_scores, labels, texts)
    all_results.append(r6)

    # ── Report ─────────────────────────────────────────────────
    rows = flatten_results(all_results)
    print_summary_table(rows)
    save_results(rows)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
