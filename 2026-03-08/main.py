#!/usr/bin/env python3
"""
Judge Reliability Harness: Stress Testing the Reliability of LLM Judges

Evaluates four reliability dimensions:
  1. Self-Consistency      – same input, multiple judge calls → agreement & variance
  2. Position Bias         – pairwise comparison with swapped order → flip rate
  3. Calibration           – known quality levels → Spearman rank correlation
  4. Perturbation Robustness – minor prompt variations → score stability

Uses real API calls to claude-haiku-4-5; no mocks or hardcoded responses.
"""

import os
import sys
import json
import time
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from anthropic import Anthropic

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
MODEL = "claude-haiku-4-5-20251001"   # fast + cheap judge
SELF_CONSISTENCY_TRIALS = 3

client = Anthropic()

# ──────────────────────────────────────────────────────────────────────────────
# Benchmark dataset  (free-response, coding, agentic)
# ──────────────────────────────────────────────────────────────────────────────
BENCHMARK = [
    # ── free-response ────────────────────────────────────────────────────────
    {
        "id": "photosynthesis",
        "format": "free-response",
        "question": "Explain what photosynthesis is.",
        "quality_order": ["excellent", "good", "poor", "wrong"],
        "responses": {
            "excellent": (
                "Photosynthesis is the process by which plants, algae, and certain bacteria "
                "convert light energy into chemical energy stored as glucose. It occurs in "
                "chloroplasts: the light-dependent reactions capture photons to produce ATP "
                "and NADPH, then the Calvin cycle uses these to fix CO₂ into organic "
                "molecules. Overall equation: 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂."
            ),
            "good": (
                "Photosynthesis is how plants make food from sunlight. They absorb CO₂ and "
                "water, use sunlight captured by chlorophyll, and produce glucose and oxygen."
            ),
            "poor": (
                "Plants do photosynthesis to get energy. They need sunlight and water. "
                "It happens in the leaves."
            ),
            "wrong": (
                "Photosynthesis is how animals digest food in their stomachs. Enzymes break "
                "down proteins and carbohydrates to release energy."
            ),
        },
    },
    # ── coding ───────────────────────────────────────────────────────────────
    {
        "id": "prime_check",
        "format": "coding",
        "question": "Write a Python function to check if a number is prime.",
        "quality_order": ["excellent", "good", "poor", "wrong"],
        "responses": {
            "excellent": (
                "def is_prime(n):\n"
                '    """Return True if n is prime, False otherwise."""\n'
                "    if n < 2:\n"
                "        return False\n"
                "    if n == 2:\n"
                "        return True\n"
                "    if n % 2 == 0:\n"
                "        return False\n"
                "    for i in range(3, int(n**0.5) + 1, 2):\n"
                "        if n % i == 0:\n"
                "            return False\n"
                "    return True"
            ),
            "good": (
                "def is_prime(n):\n"
                "    if n < 2:\n"
                "        return False\n"
                "    for i in range(2, n):\n"
                "        if n % i == 0:\n"
                "            return False\n"
                "    return True"
            ),
            "poor": "def is_prime(n):\n    return n > 1  # anything above 1 is prime",
            "wrong": "def is_prime(n):\n    return n % 2 == 0  # even numbers are prime",
        },
    },
    # ── agentic ──────────────────────────────────────────────────────────────
    {
        "id": "data_analysis_agent",
        "format": "agentic",
        "question": (
            "An AI agent was asked to: 'Analyze sales data, identify the top 3 "
            "products by revenue, and write a summary report.' "
            "Evaluate the agent's execution trace below."
        ),
        "quality_order": ["excellent", "good", "poor", "wrong"],
        "responses": {
            "excellent": (
                "Step 1 – Load data: Loaded sales_data.csv (12,450 rows, 8 columns). "
                "Verified schema: product_id, product_name, units_sold, unit_price, date.\n"
                "Step 2 – Compute revenue: revenue = units_sold × unit_price. "
                "Aggregated by product_name using groupby().sum().\n"
                "Step 3 – Rank products: Sorted descending by revenue. "
                "Top 3: Widget A ($1.2M), Gadget Pro ($980K), SmartCase ($760K).\n"
                "Step 4 – Write report: Created report.md with executive summary, "
                "revenue table, and bar chart (matplotlib). File saved successfully.\n"
                "Result: Report delivered. All steps completed with data validation."
            ),
            "good": (
                "Loaded the CSV file and calculated revenue for each product. "
                "Found the top 3 by revenue: Widget A, Gadget Pro, SmartCase. "
                "Wrote a summary with the top products and their revenues to report.txt."
            ),
            "poor": (
                "Opened the sales file. Looked at the numbers. "
                "Widget A seemed the highest. Wrote some notes."
            ),
            "wrong": (
                "I searched the internet for 'top products 2024' and listed Apple, "
                "Google, and Microsoft as the top companies. Saved to output.txt."
            ),
        },
    },
]

# Prompt prefix variants for perturbation testing
PROMPT_PREFIXES = [
    "Please evaluate the following response to the question.",
    "Assess the quality of this answer to the question.",
    "Rate this response for accuracy and completeness.",
    "Judge how well the following answer addresses the question.",
]

# ──────────────────────────────────────────────────────────────────────────────
# Core judge functions (real API calls)
# ──────────────────────────────────────────────────────────────────────────────

def ordinal_judge(question: str, response: str, system_prompt: Optional[str] = None) -> int:
    """Score a response 1-5. Returns integer."""
    system = system_prompt or (
        "You are an expert evaluator. Given a question and a response, "
        "score the response on a scale of 1 to 5:\n"
        "1 = Completely wrong or irrelevant\n"
        "2 = Mostly incorrect with minor correct elements\n"
        "3 = Partially correct but incomplete\n"
        "4 = Mostly correct with minor gaps\n"
        "5 = Excellent: accurate, complete, and clear\n"
        "Reply with ONLY the digit (1, 2, 3, 4, or 5). No other text."
    )
    msg = client.messages.create(
        model=MODEL,
        max_tokens=5,
        system=system,
        messages=[{"role": "user", "content": f"Question: {question}\n\nResponse: {response}"}],
    )
    text = msg.content[0].text.strip()
    for ch in text:
        if ch in "12345":
            return int(ch)
    return 3  # fallback to middle if parse fails


def binary_judge(question: str, response: str, system_prompt: Optional[str] = None) -> str:
    """Binary judge: returns 'good' or 'bad'."""
    system = system_prompt or (
        "You are an expert evaluator. Given a question and a response, "
        "determine if the response is GOOD (accurate, helpful, reasonably complete) "
        "or BAD (inaccurate, misleading, or unhelpful). "
        "Reply with exactly one word: 'good' or 'bad'. No other text."
    )
    msg = client.messages.create(
        model=MODEL,
        max_tokens=5,
        system=system,
        messages=[{"role": "user", "content": f"Question: {question}\n\nResponse: {response}"}],
    )
    text = msg.content[0].text.strip().lower()
    return "good" if "good" in text else "bad"


def pairwise_judge(question: str, response_a: str, response_b: str) -> str:
    """Pairwise judge: returns 'A', 'B', or 'tie'."""
    system = (
        "You are an expert evaluator. Given a question and two responses labeled A and B, "
        "decide which is better overall. "
        "Reply with exactly one token: 'A', 'B', or 'tie'. No other text."
    )
    content = (
        f"Question: {question}\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}"
    )
    msg = client.messages.create(
        model=MODEL,
        max_tokens=5,
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    text = msg.content[0].text.strip().upper()
    if "TIE" in text:
        return "tie"
    if "A" in text and "B" not in text:
        return "A"
    if "B" in text and "A" not in text:
        return "B"
    return "tie"


# ──────────────────────────────────────────────────────────────────────────────
# Reliability tests
# ──────────────────────────────────────────────────────────────────────────────

def test_self_consistency() -> pd.DataFrame:
    """
    Same (question, response) pair judged SELF_CONSISTENCY_TRIALS times.
    Measures: ordinal score variance, binary agreement rate.
    """
    print(f"\n{'='*60}")
    print(f"TEST 1/4: Self-Consistency  ({SELF_CONSISTENCY_TRIALS} trials per item)")
    print("="*60)

    rows = []
    total = len(BENCHMARK) * len(next(iter(BENCHMARK))["responses"])
    done = 0

    for item in BENCHMARK:
        qid, question = item["id"], item["question"]
        for quality, response in item["responses"].items():
            done += 1
            scores, judgments = [], []
            for t in range(SELF_CONSISTENCY_TRIALS):
                s = ordinal_judge(question, response)
                j = binary_judge(question, response)
                scores.append(s)
                judgments.append(j)
                sys.stdout.write(
                    f"  [{done:02d}/{total}] {qid:<22} {quality:<10} "
                    f"trial {t+1}: ordinal={s}  binary={j}\n"
                )
                sys.stdout.flush()

            agree = judgments.count(judgments[0]) / len(judgments)
            rows.append({
                "item_id": qid, "format": item["format"], "quality": quality,
                "ordinal_scores": scores,
                "score_mean": round(np.mean(scores), 2),
                "score_std":  round(np.std(scores),  3),
                "score_var":  round(np.var(scores),  3),
                "binary_judgments": judgments,
                "binary_agree": round(agree, 3),
            })

    df = pd.DataFrame(rows)
    print(f"\n  Summary:")
    print(f"    Avg ordinal score std   : {df['score_std'].mean():.3f}  (lower = more consistent)")
    print(f"    Avg binary agree rate   : {df['binary_agree'].mean():.1%}")
    print(f"    Items fully binary-agree: {(df['binary_agree'] == 1.0).sum()}/{len(df)}")
    return df


def test_position_bias() -> pd.DataFrame:
    """
    Present response pair (higher vs lower quality) in both orders.
    Measures: preference flip rate when order is reversed.
    """
    print(f"\n{'='*60}")
    print("TEST 2/4: Position Bias")
    print("="*60)

    rows = []
    pairs = [("excellent", "poor"), ("good", "wrong")]

    for item in BENCHMARK:
        qid, question = item["id"], item["question"]
        for qa, qb in pairs:
            ra = item["responses"][qa]
            rb = item["responses"][qb]

            # Forward: A = higher quality
            fwd = pairwise_judge(question, ra, rb)
            # Reversed: A = lower quality
            rev = pairwise_judge(question, rb, ra)

            # Consistent means: forward prefers A (higher) AND reversed prefers B (higher)
            # or forward=tie and reversed=tie
            consistent = (
                (fwd == "A" and rev == "B") or
                (fwd == "B" and rev == "A") or
                (fwd == "tie" and rev == "tie")
            )
            fwd_correct = fwd == "A"   # correct: higher quality is A
            rev_correct = rev == "B"   # correct: higher quality is now B

            print(
                f"  {qid:<22} {qa:10} vs {qb:10} │ "
                f"fwd={fwd}  rev={rev}  consistent={consistent}"
            )
            rows.append({
                "item_id": qid, "format": item["format"],
                "pair": f"{qa} vs {qb}",
                "forward_winner": fwd,
                "reversed_winner": rev,
                "fwd_correct": fwd_correct,
                "rev_correct": rev_correct,
                "position_consistent": consistent,
            })

    df = pd.DataFrame(rows)
    bias_rate = 1 - df["position_consistent"].mean()
    print(f"\n  Summary:")
    print(f"    Position bias rate       : {bias_rate:.1%}  (lower = less biased)")
    print(f"    Forward accuracy         : {df['fwd_correct'].mean():.1%}")
    print(f"    Reversed accuracy        : {df['rev_correct'].mean():.1%}")
    return df


def test_calibration() -> pd.DataFrame:
    """
    Score all four quality levels (excellent/good/poor/wrong).
    Measures: Spearman correlation between expected rank and actual score,
    and whether ordering is monotonically correct.
    """
    print(f"\n{'='*60}")
    print("TEST 3/4: Calibration (Spearman rank correlation)")
    print("="*60)

    expected_rank = {"excellent": 4, "good": 3, "poor": 2, "wrong": 1}
    rows = []

    for item in BENCHMARK:
        qid, question = item["id"], item["question"]
        actual_scores = {}
        for quality, response in item["responses"].items():
            score = ordinal_judge(question, response)
            actual_scores[quality] = score
            print(f"  {qid:<22} {quality:<10} → score={score}  (expected rank {expected_rank[quality]})")

        exp_vals = [expected_rank[q] for q in item["quality_order"]]
        act_vals = [actual_scores[q] for q in item["quality_order"]]

        if len(set(act_vals)) > 1:
            corr, pval = spearmanr(exp_vals, act_vals)
        else:
            corr, pval = 0.0, 1.0

        monotonic = all(act_vals[i] >= act_vals[i + 1] for i in range(len(act_vals) - 1))

        rows.append({
            "item_id": qid, "format": item["format"],
            "expected_ranks": exp_vals,
            "actual_scores":  act_vals,
            "spearman_corr":  round(float(corr), 3),
            "spearman_pval":  round(float(pval), 4),
            "monotonic":      monotonic,
        })
        print(f"    Spearman r={corr:.3f}  p={pval:.4f}  monotonic={monotonic}")

    df = pd.DataFrame(rows)
    print(f"\n  Summary:")
    print(f"    Avg Spearman correlation : {df['spearman_corr'].mean():.3f}  (1.0 = perfect)")
    print(f"    Monotonic ordering rate  : {df['monotonic'].mean():.1%}")
    return df


def test_perturbation_robustness() -> pd.DataFrame:
    """
    Judge with 4 different prompt-prefix variants.
    Measures: score std and range across perturbations.
    """
    print(f"\n{'='*60}")
    print("TEST 4/4: Perturbation Robustness (prompt prefix variants)")
    print("="*60)

    rows = []
    # Test on 'good' and 'poor' responses for first 2 benchmark items
    for item in BENCHMARK[:2]:
        qid, question = item["id"], item["question"]
        for quality in ("good", "poor"):
            response = item["responses"][quality]
            scores = []
            for i, prefix in enumerate(PROMPT_PREFIXES):
                system = (
                    f"{prefix}\n\n"
                    "Score 1-5 where:\n"
                    "1=completely wrong, 2=mostly wrong, 3=partially correct, "
                    "4=mostly correct, 5=excellent.\n"
                    "Reply with ONLY the digit."
                )
                score = ordinal_judge(question, response, system_prompt=system)
                scores.append(score)
                print(f"  {qid:<22} {quality:<6} prefix {i+1}: score={score}")

            rows.append({
                "item_id": qid, "format": item["format"],
                "quality": quality,
                "perturbed_scores": scores,
                "score_mean": round(np.mean(scores), 2),
                "score_std":  round(np.std(scores),  3),
                "score_range": int(max(scores) - min(scores)),
            })

    df = pd.DataFrame(rows)
    print(f"\n  Summary:")
    print(f"    Avg score std under perturbation : {df['score_std'].mean():.3f}")
    print(f"    Avg score range under perturbation: {df['score_range'].mean():.2f}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Aggregate report
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_report(sc_df, pb_df, cal_df, rob_df) -> dict:
    """Compute overall reliability scores and return as dict."""

    # 1. Self-consistency score (0-1, higher = more consistent)
    #    Based on binary agreement rate and inverse of normalised variance
    binary_agree    = sc_df["binary_agree"].mean()
    norm_variance   = sc_df["score_var"].mean() / 4.0   # max ordinal var ~4
    ordinal_consist = 1.0 - norm_variance
    consistency_score = 0.5 * binary_agree + 0.5 * ordinal_consist

    # 2. Position bias score (0-1, higher = less biased)
    position_score = float(pb_df["position_consistent"].mean())

    # 3. Calibration score (0-1, based on Spearman corr shifted to [0,1])
    calibration_score = float((cal_df["spearman_corr"].mean() + 1) / 2)

    # 4. Robustness score (0-1, higher = more robust)
    max_possible_range = 4.0
    robustness_score = 1.0 - float(rob_df["score_range"].mean()) / max_possible_range

    # Composite
    composite = np.mean([consistency_score, position_score, calibration_score, robustness_score])

    return {
        "self_consistency":  round(consistency_score, 3),
        "position_fairness": round(position_score, 3),
        "calibration":       round(calibration_score, 3),
        "perturbation_robustness": round(robustness_score, 3),
        "composite_reliability":   round(float(composite), 3),
    }


def print_final_report(scores: dict, elapsed: float):
    WIDTH = 62
    print("\n" + "=" * WIDTH)
    print("  JUDGE RELIABILITY HARNESS — FINAL REPORT")
    print("=" * WIDTH)
    print(f"  Judge model   : {MODEL}")
    print(f"  Benchmark size: {len(BENCHMARK)} items  "
          f"({sum(len(i['responses']) for i in BENCHMARK)} response variants)")
    print(f"  Elapsed time  : {elapsed:.1f}s")
    print("-" * WIDTH)
    labels = {
        "self_consistency":        "Self-Consistency    (agree across trials)",
        "position_fairness":       "Position Fairness   (pairwise order swap)",
        "calibration":             "Calibration         (Spearman rank corr.)",
        "perturbation_robustness": "Perturbation Robust (prompt variation)",
        "composite_reliability":   "── COMPOSITE RELIABILITY ──",
    }
    for key, label in labels.items():
        val   = scores[key]
        bar   = "█" * int(val * 30) + "░" * (30 - int(val * 30))
        grade = "PASS" if val >= 0.70 else ("WARN" if val >= 0.50 else "FAIL")
        if key == "composite_reliability":
            print("-" * WIDTH)
        print(f"  {label:<42} {val:.3f}  [{grade}]")
        print(f"    {bar}")
    print("=" * WIDTH)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       Judge Reliability Harness — LLM Judge Stress Test  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Judge model   : {MODEL}")
    print(f"  Benchmark     : {len(BENCHMARK)} items across free-response / coding / agentic")
    print(f"  Trials        : {SELF_CONSISTENCY_TRIALS} per item (self-consistency test)")
    print()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    t0 = time.time()

    sc_df  = test_self_consistency()
    pb_df  = test_position_bias()
    cal_df = test_calibration()
    rob_df = test_perturbation_robustness()

    elapsed = time.time() - t0

    scores = aggregate_report(sc_df, pb_df, cal_df, rob_df)
    print_final_report(scores, elapsed)

    # Save detailed results
    out = {
        "config": {"model": MODEL, "benchmark_size": len(BENCHMARK), "trials": SELF_CONSISTENCY_TRIALS},
        "scores": scores,
        "self_consistency": sc_df.drop(columns=["ordinal_scores", "binary_judgments"]).to_dict(orient="records"),
        "position_bias":    pb_df.to_dict(orient="records"),
        "calibration":      cal_df.to_dict(orient="records"),
        "robustness":       rob_df.drop(columns=["perturbed_scores"]).to_dict(orient="records"),
    }
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Detailed results saved to results.json")


if __name__ == "__main__":
    main()
