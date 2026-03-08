#!/usr/bin/env python3
"""
Judge Reliability Harness
Stress testing the reliability of LLM judges across:
  1. Intra-judge consistency  (same input → same verdict across trials)
  2. Positional bias          (swapping A/B shouldn't change preference)
  3. Calibration accuracy     (agreement with human ground truth)
  4. Ordinal rating stability (repeat scalar ratings stay correlated)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from anthropic import Anthropic
from datasets import load_dataset
from sklearn.metrics import accuracy_score, cohen_kappa_score
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
JUDGE_MODEL = "claude-haiku-4-5-20251001"

N_CONSISTENCY_SAMPLES = 5
N_CONSISTENCY_TRIALS  = 3   # repeat each judgment 3× to check stability

N_POSITIONAL_SAMPLES  = 5   # swap A↔B and check verdict flips

N_CALIBRATION_SAMPLES = 10  # compare judge to human annotations

N_ORDINAL_SAMPLES     = 5
N_ORDINAL_TRIALS      = 2   # repeat ordinal ratings 2×

MAX_POST_CHARS        = 400  # truncate long posts for speed
MAX_SUMMARY_CHARS     = 200

client = Anthropic()


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
def _fallback_samples(n: int) -> list[dict]:
    """Return synthetic summarisation pairs when the HF dataset is unavailable."""
    base = [
        {
            "question": "My landlord refuses to fix the heating in my apartment despite repeated requests over 3 months. It's now winter and the temperature inside is dangerously low. I've documented everything in writing. What are my options?",
            "response_a": "You have several legal options. Send a formal written notice to your landlord citing the habitability violation. Contact your local housing authority or tenant rights organization. You may be able to withhold rent, repair-and-deduct, or terminate the lease depending on your jurisdiction. Consider small claims court for damages.",
            "response_b": "Talk to your landlord again. If that doesn't work, you could try calling a local housing agency. Document everything and maybe consult a lawyer if things don't improve soon.",
            "winner": "A",
        },
        {
            "question": "I'm trying to decide between learning Python or JavaScript as my first programming language. I want to eventually work in web development but also interested in data science.",
            "response_a": "Learn Python first. It has cleaner syntax for beginners, dominates data science, and is widely used in web backends (Django/Flask). You can pick up JavaScript later for front-end work, and the transition will be easier once you understand programming fundamentals.",
            "response_b": "Both languages are great. Python is good for data science while JavaScript is essential for web. You should consider what projects interest you most and start there. Many resources exist for both.",
            "winner": "A",
        },
        {
            "question": "What are the main differences between supervised and unsupervised machine learning?",
            "response_a": "The difference is in whether you have labels. Supervised learning uses labeled data to train models that predict outputs, like classification or regression. Unsupervised learning finds patterns in unlabeled data, like clustering or dimensionality reduction.",
            "response_b": "Supervised learning is when the algorithm learns from labeled training data, mapping inputs to known outputs (e.g., spam detection, image classification). Unsupervised learning discovers hidden patterns or structure in unlabeled data without predefined answers (e.g., customer segmentation, anomaly detection). The key distinction is the presence or absence of ground-truth labels during training.",
            "winner": "B",
        },
        {
            "question": "I've been feeling very anxious lately and it's affecting my work and relationships. I haven't tried anything yet. What should I do first?",
            "response_a": "Start by talking to your primary care doctor or a mental health professional. They can assess your symptoms and recommend appropriate treatment, which may include therapy (CBT is highly effective for anxiety), lifestyle changes, or medication if needed. In the meantime, try regular exercise, limiting caffeine and alcohol, and practicing mindfulness.",
            "response_b": "You should definitely see a therapist. Anxiety is very common and treatable. Try to relax and don't stress too much about it. Exercise and sleep also help with anxiety symptoms.",
            "winner": "A",
        },
        {
            "question": "Can you explain how HTTPS works in simple terms?",
            "response_a": "HTTPS encrypts data between your browser and a website using TLS. The site sends a certificate proving its identity, your browser verifies it, and they establish an encrypted channel so no one in between can read your data.",
            "response_b": "HTTPS is the secure version of HTTP. It uses TLS (Transport Layer Security) to encrypt communications. When you connect, the server presents a digital certificate verified by a trusted Certificate Authority. Your browser and server then perform a handshake to establish encryption keys, after which all data is encrypted in transit, preventing eavesdropping and tampering.",
            "winner": "B",
        },
        {
            "question": "My team consistently misses deadlines. As a new manager, how should I address this?",
            "response_a": "First, understand why deadlines are being missed—through 1-on-1s and team retrospectives. Common causes include unclear scope, unrealistic estimates, or blockers. Then improve processes: clearer requirements, realistic timelines set collaboratively, regular check-ins, and removing obstacles. Address accountability without blame.",
            "response_b": "Hold a team meeting to discuss the issue. Set clear expectations going forward and make sure everyone knows the consequences of missing deadlines. Track progress more closely and consider implementing project management tools.",
            "winner": "A",
        },
        {
            "question": "What's the difference between RAM and storage (SSD/HDD)?",
            "response_a": "RAM is temporary fast memory used while your computer runs programs. Storage (SSD/HDD) holds data permanently. RAM loses its contents when powered off; storage keeps files indefinitely. More RAM lets you run more programs simultaneously; more storage means more files.",
            "response_b": "RAM (Random Access Memory) is your computer's short-term working memory—fast but volatile, holding data only while powered on. Storage (SSD/HDD) is long-term, persistent memory for files and programs. When you open an app, it loads from storage into RAM for fast access. RAM speed matters for multitasking; storage capacity matters for how much you can save.",
            "winner": "B",
        },
        {
            "question": "I want to start investing but I have no experience. Where should I begin?",
            "response_a": "Start by building an emergency fund (3-6 months expenses), then invest in low-cost index funds through a tax-advantaged account (401k/IRA). A simple three-fund portfolio (US stocks, international stocks, bonds) matches your risk tolerance. Avoid individual stock picking until you have more experience.",
            "response_b": "Investing can seem intimidating but it's important to start. Consider your goals and risk tolerance first. Index funds are good for beginners. Make sure to diversify and think long-term. You might want to consult a financial advisor.",
            "winner": "A",
        },
        {
            "question": "How does photosynthesis work?",
            "response_a": "Photosynthesis converts sunlight, water, and CO2 into glucose and oxygen. Plants use chlorophyll in two stages: light reactions capture energy from sunlight, and the Calvin cycle uses that energy to build glucose from CO2.",
            "response_b": "Photosynthesis is the process plants use to make food from light. It happens in chloroplasts using chlorophyll. The overall equation is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2. The light-dependent reactions produce ATP and NADPH; the Calvin cycle uses these to fix carbon dioxide into glucose.",
            "winner": "B",
        },
        {
            "question": "What are some effective strategies for learning a new language?",
            "response_a": "Immersion and consistency are key: daily practice (even 15-20 minutes), consuming media in the target language, speaking with natives via apps like iTalki, using spaced repetition for vocabulary (Anki), and focusing on high-frequency words first. Grammar should be learned in context rather than in isolation.",
            "response_b": "Practice every day and try to immerse yourself in the language. Watch movies, listen to music, and find a language partner. Use apps like Duolingo. Don't be afraid to make mistakes as that's how you learn.",
            "winner": "A",
        },
    ]
    # Cycle through base samples if more are needed
    result = []
    while len(result) < n:
        result.extend(base)
    return result[:n]


def load_data(n: int) -> list[dict]:
    """Stream human-preference pairs from openai/summarize_from_feedback."""
    print("[1/5] Loading benchmark dataset (openai/summarize_from_feedback)…")
    try:
        ds = load_dataset(
            "openai/summarize_from_feedback",
            "comparisons",
            split="train",
            streaming=True,
        )

        samples = []
        for item in ds:
            post   = (item["info"]["post"] or "")[:MAX_POST_CHARS].strip()
            sum_a  = (item["summaries"][0]["text"] or "")[:MAX_SUMMARY_CHARS].strip()
            sum_b  = (item["summaries"][1]["text"] or "")[:MAX_SUMMARY_CHARS].strip()
            choice = item["choice"]           # 0 = A preferred, 1 = B preferred

            if not post or not sum_a or not sum_b:
                continue

            samples.append({
                "question":   post,   # raw post text; judge prompts add the task framing
                "response_a": sum_a,
                "response_b": sum_b,
                "winner":     "A" if choice == 0 else "B",
            })

            if len(samples) >= n:
                break

        if samples:
            print(f"  Loaded {len(samples)} samples.\n")
            return samples
    except Exception as e:
        print(f"  Dataset unavailable ({e}); using synthetic fallback data.")

    samples = _fallback_samples(n)
    print(f"  Loaded {len(samples)} synthetic samples.\n")
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Judge primitives
# ──────────────────────────────────────────────────────────────────────────────
def judge_pairwise(question: str, response_a: str, response_b: str) -> str:
    """Return 'A' or 'B' — which summary is better."""
    prompt = (
        f"You are an expert evaluator assessing summarisation quality.\n\n"
        f"Original post:\n{question}\n\n"
        f"Summary A:\n{response_a}\n\n"
        f"Summary B:\n{response_b}\n\n"
        f"Which summary is better overall (accuracy, conciseness, coverage)?\n"
        f"Reply with ONLY the single letter A or B."
    )
    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=5,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip().upper()
    return "A" if "A" in text else "B"


def judge_ordinal(question: str, response: str, scale: int = 10) -> int:
    """Return integer rating 1–scale for a single response."""
    prompt = (
        f"You are an expert evaluator.\n\n"
        f"Original post:\n{question}\n\n"
        f"Summary:\n{response}\n\n"
        f"Rate the summary quality from 1 (very poor) to {scale} (excellent).\n"
        f"Consider accuracy, conciseness, and coverage.\n"
        f"Reply with ONLY a single integer from 1 to {scale}."
    )
    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=5,
        messages=[{"role": "user", "content": prompt}],
    )
    digits = "".join(filter(str.isdigit, msg.content[0].text.strip()))
    try:
        return max(1, min(scale, int(digits)))
    except ValueError:
        return scale // 2


# ──────────────────────────────────────────────────────────────────────────────
# Reliability tests
# ──────────────────────────────────────────────────────────────────────────────
def test_consistency(samples: list[dict]) -> tuple[list[dict], float]:
    """Intra-judge consistency: repeat each comparison N times."""
    print(f"[2/5] Consistency Test  ({N_CONSISTENCY_TRIALS} trials × "
          f"{N_CONSISTENCY_SAMPLES} samples)…")
    rows = []
    for i, s in enumerate(samples[:N_CONSISTENCY_SAMPLES]):
        verdicts = [
            judge_pairwise(s["question"], s["response_a"], s["response_b"])
            for _ in range(N_CONSISTENCY_TRIALS)
        ]
        majority   = max(set(verdicts), key=verdicts.count)
        cons_rate  = verdicts.count(majority) / len(verdicts)
        rows.append({"sample": i + 1, "verdicts": verdicts,
                     "majority": majority, "consistency_rate": cons_rate})
        print(f"  sample {i+1}: {verdicts}  →  {cons_rate:.0%} consistent")

    avg = float(np.mean([r["consistency_rate"] for r in rows]))
    print(f"  → Average intra-judge consistency: {avg:.1%}\n")
    return rows, avg


def test_positional_bias(samples: list[dict]) -> tuple[list[dict], float]:
    """Positional bias: does swapping A↔B flip the verdict?"""
    print(f"[3/5] Positional Bias Test  ({N_POSITIONAL_SAMPLES} samples)…")
    rows = []
    for i, s in enumerate(samples[:N_POSITIONAL_SAMPLES]):
        v_ab = judge_pairwise(s["question"], s["response_a"], s["response_b"])
        v_ba = judge_pairwise(s["question"], s["response_b"], s["response_a"])
        # normalise: in BA order, "A" means original B was preferred
        norm_ba     = "B" if v_ba == "A" else "A"
        invariant   = (v_ab == norm_ba)
        rows.append({"sample": i + 1, "verdict_AB": v_ab,
                     "verdict_BA_norm": norm_ba, "position_invariant": invariant})
        tag = "✓" if invariant else "✗ BIAS"
        print(f"  sample {i+1}: AB={v_ab}  BA(norm)={norm_ba}  {tag}")

    bias_rate = 1.0 - float(np.mean([r["position_invariant"] for r in rows]))
    print(f"  → Positional bias rate: {bias_rate:.1%}\n")
    return rows, bias_rate


def test_calibration(samples: list[dict]) -> tuple[list, list, float, float]:
    """Calibration: judge accuracy vs. human preference labels."""
    print(f"[4/5] Calibration Test  ({N_CALIBRATION_SAMPLES} samples vs. human labels)…")
    preds, truths = [], []
    for i, s in enumerate(samples[:N_CALIBRATION_SAMPLES]):
        pred  = judge_pairwise(s["question"], s["response_a"], s["response_b"])
        truth = s["winner"]
        preds.append(pred)
        truths.append(truth)
        tag = "✓" if pred == truth else "✗"
        print(f"  sample {i+1}: judge={pred}  human={truth}  {tag}")

    acc = accuracy_score(truths, preds)
    try:
        kappa = cohen_kappa_score(truths, preds) if len(set(truths)) > 1 else float("nan")
    except Exception:
        kappa = float("nan")

    print(f"  → Accuracy: {acc:.1%}   Cohen's κ: {kappa:.3f}\n")
    return preds, truths, acc, kappa


def test_ordinal_reliability(samples: list[dict]) -> tuple[list[list], float]:
    """Ordinal stability: repeat scalar ratings and measure correlation."""
    print(f"[5/5] Ordinal Reliability Test  ({N_ORDINAL_TRIALS} trials × "
          f"{N_ORDINAL_SAMPLES} samples)…")
    trials: list[list[int]] = [[] for _ in range(N_ORDINAL_TRIALS)]

    for i, s in enumerate(samples[:N_ORDINAL_SAMPLES]):
        trial_pairs = []
        for t in range(N_ORDINAL_TRIALS):
            ra = judge_ordinal(s["question"], s["response_a"])
            rb = judge_ordinal(s["question"], s["response_b"])
            trials[t].extend([ra, rb])
            trial_pairs.append(f"T{t+1}:(A={ra},B={rb})")
        print(f"  sample {i+1}: {' '.join(trial_pairs)}")

    # Spearman ρ between each pair of trial vectors
    corrs = []
    for t1 in range(N_ORDINAL_TRIALS):
        for t2 in range(t1 + 1, N_ORDINAL_TRIALS):
            r, _ = spearmanr(trials[t1], trials[t2])
            corrs.append(r)

    avg_corr = float(np.nanmean(corrs)) if corrs else float("nan")
    print(f"  → Inter-trial Spearman ρ: {avg_corr:.3f}\n")
    return trials, avg_corr


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────
def print_report(avg_consistency, bias_rate, accuracy, kappa, avg_corr):
    """Print summary table and save results.json."""
    # Composite reliability score (weighted average of sub-metrics)
    sub_scores = [avg_consistency, 1.0 - bias_rate, accuracy, max(0.0, avg_corr)]
    weights    = [0.30,             0.20,             0.30,    0.20]
    score      = float(np.dot(sub_scores, weights))
    grade      = "HIGH" if score > 0.75 else ("MEDIUM" if score > 0.50 else "LOW")

    df = pd.DataFrame([
        {"Metric": "Intra-judge consistency", "Value": f"{avg_consistency:.1%}",
         "Interpretation": "higher → more reliable"},
        {"Metric": "Positional bias rate",    "Value": f"{bias_rate:.1%}",
         "Interpretation": "lower → less biased"},
        {"Metric": "Calibration accuracy",    "Value": f"{accuracy:.1%}",
         "Interpretation": "vs. human annotations"},
        {"Metric": "Cohen's κ",               "Value": f"{kappa:.3f}",
         "Interpretation": ">0.60 = substantial agreement"},
        {"Metric": "Ordinal Spearman ρ",      "Value": f"{avg_corr:.3f}",
         "Interpretation": ">0.80 = high ordinal stability"},
    ])

    print("=" * 62)
    print("  JUDGE RELIABILITY REPORT")
    print("=" * 62)
    print(f"  Model  : {JUDGE_MODEL}")
    print()
    print(df.to_string(index=False))
    print()
    print(f"  Overall Reliability Score : {score:.1%}  [{grade}]")
    print("=" * 62)

    results = {
        "judge_model": JUDGE_MODEL,
        "metrics": {
            "intra_judge_consistency": f"{avg_consistency:.1%}",
            "positional_bias_rate":    f"{bias_rate:.1%}",
            "calibration_accuracy":    f"{accuracy:.1%}",
            "cohen_kappa":             f"{kappa:.3f}",
            "ordinal_spearman_rho":    f"{avg_corr:.3f}",
        },
        "overall_reliability_score": f"{score:.1%}",
        "grade": grade,
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Saved → results.json")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  JUDGE RELIABILITY HARNESS")
    print("  Stress-testing LLM judge reliability")
    print("=" * 62)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    t0 = time.time()

    # Need enough samples for all tests
    n_needed = max(N_CONSISTENCY_SAMPLES, N_POSITIONAL_SAMPLES,
                   N_CALIBRATION_SAMPLES, N_ORDINAL_SAMPLES)
    samples = load_data(n_needed)

    if len(samples) < n_needed:
        print(f"WARNING: only {len(samples)} samples loaded; "
              f"some tests may use fewer samples.")

    _, avg_consistency = test_consistency(samples)
    _, bias_rate       = test_positional_bias(samples)
    _, _, accuracy, kappa = test_calibration(samples)
    _, avg_corr        = test_ordinal_reliability(samples)

    print(f"\n  Total wall-clock time: {time.time() - t0:.1f}s\n")
    print_report(avg_consistency, bias_rate, accuracy, kappa, avg_corr)


if __name__ == "__main__":
    main()
