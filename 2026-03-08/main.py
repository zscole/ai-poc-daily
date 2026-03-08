#!/usr/bin/env python3
"""
NCTB-QA: Bangla Educational Question Answering POC

Demonstrates fine-tuning XLM-RoBERTa on Bangla QA data (BnQUAD / NCTB-QA)
with support for unanswerable questions (SQuAD 2.0 style).

Key contributions of NCTB-QA paper reproduced here:
  - Extractive QA over Bangla educational texts
  - Unanswerable question detection (null-answer threshold)
  - Evaluation with Exact Match and F1 scores
"""

import collections
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    default_data_collator,
    get_linear_schedule_with_warmup,
)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "xlm-roberta-base"       # multilingual, handles Bangla well
MAX_LENGTH   = 384
DOC_STRIDE   = 128
BATCH_SIZE   = 8
NUM_EPOCHS   = 3
TRAIN_LIMIT  = 300   # examples for a quick POC run
EVAL_LIMIT   = 100
LR           = 2e-5
NULL_SCORE_DIFF_THRESHOLD = 0.0   # tune for unanswerable detection

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── 1. Data loading ────────────────────────────────────────────────────────────

def load_bangla_qa() -> DatasetDict:
    """Load a real Bangla QA dataset. Try NCTB-QA, fall back to BnQUAD."""
    candidates = [
        ("Tahsin-Mayeesha/nctb-qa",   "NCTB-QA"),
        ("csebuetnlp/bnquad",          "BnQUAD"),
    ]
    for repo, name in candidates:
        try:
            print(f"Trying to load {name} ({repo}) …")
            ds = load_dataset(repo, trust_remote_code=True)
            print(f"✓ Loaded {name}: {ds}")
            return ds, name
        except Exception as e:
            print(f"  ✗ {e}")

    raise RuntimeError("Could not load any Bangla QA dataset. Check internet / HuggingFace.")


def normalise_squad_example(ex: dict) -> dict:
    """Ensure each example has the SQuAD 2.0 fields we need."""
    # BnQUAD uses 'answers' as {'text': [...], 'answer_start': [...]}
    answers = ex.get("answers", {})
    if isinstance(answers, dict):
        texts   = answers.get("text",         answers.get("answer_text", []))
        starts  = answers.get("answer_start",  answers.get("answer_start_token", []))
    else:
        texts, starts = [], []

    return {
        "id":       str(ex.get("id", ex.get("qas_id", ""))),
        "context":  ex.get("context", ex.get("passage", "")),
        "question": ex.get("question", ""),
        "answers":  {"text": texts, "answer_start": starts},
        "is_impossible": len(texts) == 0,
    }


# ── 2. Tokenisation helpers ────────────────────────────────────────────────────

def preprocess_train(examples, tokenizer):
    """Tokenise training examples; map each token back to its answer span."""
    questions = [q.lstrip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map   = inputs.pop("overflow_to_sample_mapping")
    offset_map   = inputs.pop("offset_mapping")

    start_positions, end_positions = [], []

    for i, offsets in enumerate(offset_map):
        sample_idx  = sample_map[i]
        answers     = examples["answers"][sample_idx]
        cls_index   = inputs["input_ids"][i].index(tokenizer.cls_token_id)

        seq_ids = inputs.sequence_ids(i)
        ctx_start = next(j for j, s in enumerate(seq_ids) if s == 1)
        ctx_end   = len(seq_ids) - 1 - next(
            j for j, s in enumerate(reversed(seq_ids)) if s == 1
        )

        # Unanswerable
        if len(answers["text"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        ans_start_char = answers["answer_start"][0]
        ans_end_char   = ans_start_char + len(answers["text"][0])

        # Answer falls outside this chunk → mark as unanswerable in this chunk
        if (offsets[ctx_start][0] > ans_end_char or
                offsets[ctx_end][1] < ans_start_char):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            # Walk to first token whose offset starts >= ans_start_char
            tok_start = ctx_start
            while tok_start <= ctx_end and offsets[tok_start][0] < ans_start_char:
                tok_start += 1
            tok_end = ctx_end
            while tok_end >= ctx_start and offsets[tok_end][1] > ans_end_char:
                tok_end -= 1
            start_positions.append(tok_start)
            end_positions.append(tok_end)

    inputs["start_positions"] = start_positions
    inputs["end_positions"]   = end_positions
    return inputs


def preprocess_eval(examples, tokenizer):
    """Tokenise validation examples (keep offset map for answer extraction)."""
    questions = [q.lstrip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []
    for i in range(len(inputs["input_ids"])):
        example_ids.append(examples["id"][sample_map[i]])
        seq_ids = inputs.sequence_ids(i)
        # Use (-1, -1) sentinel for non-context tokens; Arrow can't store None in int lists
        inputs["offset_mapping"][i] = [
            (o[0], o[1]) if seq_ids[j] == 1 else (-1, -1)
            for j, o in enumerate(inputs["offset_mapping"][i])
        ]
    inputs["example_id"] = example_ids
    return inputs


# ── 3. Answer postprocessing (SQuAD 2.0) ──────────────────────────────────────

def postprocess_qa_predictions(examples, features, raw_logits,
                                n_best=20, max_answer_len=30,
                                null_thresh=NULL_SCORE_DIFF_THRESHOLD):
    all_start_logits, all_end_logits = raw_logits

    example_id_to_index = {ex["id"]: i for i, ex in enumerate(examples)}
    features_per_example = collections.defaultdict(list)
    for i, feat_id in enumerate(features["example_id"]):
        features_per_example[example_id_to_index[feat_id]].append(i)

    predictions, references = {}, {}

    for ex_idx, example in enumerate(examples):
        feat_indices   = features_per_example[ex_idx]
        min_null_score = None
        valid_answers  = []

        context = example["context"]
        for feat_idx in feat_indices:
            start_logits = all_start_logits[feat_idx]
            end_logits   = all_end_logits[feat_idx]
            offsets      = features["offset_mapping"][feat_idx]

            # Score for null answer (CLS token at index 0)
            null_score = start_logits[0] + end_logits[0]
            if min_null_score is None or null_score < min_null_score:
                min_null_score = null_score

            start_indexes = np.argsort(start_logits)[-1 : -n_best - 1 : -1].tolist()
            end_indexes   = np.argsort(end_logits)[-1 : -n_best - 1 : -1].tolist()

            for si in start_indexes:
                for ei in end_indexes:
                    # (-1, -1) sentinel means non-context token
                    if offsets[si][0] == -1 or offsets[ei][0] == -1:
                        continue
                    if ei < si or ei - si + 1 > max_answer_len:
                        continue
                    valid_answers.append({
                        "score": start_logits[si] + end_logits[ei],
                        "text":  context[offsets[si][0]: offsets[ei][1]],
                    })

        if valid_answers:
            best = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            if min_null_score - best["score"] > null_thresh:
                predictions[example["id"]] = ""          # unanswerable
            else:
                predictions[example["id"]] = best["text"]
        else:
            predictions[example["id"]] = ""

        # Ground-truth reference
        references[example["id"]] = {
            "answers": example["answers"],
            "id":      example["id"],
        }

    return predictions, references


# ── 4. Metrics ────────────────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    import re, string
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    return ' '.join(s.split())


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def compute_metrics(predictions: dict, references: dict):
    em_scores, f1_scores = [], []
    for qid, pred_text in predictions.items():
        ref = references[qid]
        gold_answers = ref["answers"]["text"]
        if not gold_answers:                     # truly unanswerable
            em_scores.append(float(pred_text == ""))
            f1_scores.append(float(pred_text == ""))
        else:
            em_scores.append(max(exact_match(pred_text, g) for g in gold_answers))
            f1_scores.append(max(f1_score(pred_text, g) for g in gold_answers))
    return {
        "exact_match": np.mean(em_scores) * 100,
        "f1":          np.mean(f1_scores) * 100,
    }


# ── 5. Training loop ──────────────────────────────────────────────────────────

def train(model, train_loader, optimizer, scheduler, epoch: int):
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss    = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        if (step + 1) % 10 == 0 or step == 0:
            avg = total_loss / (step + 1)
            print(f"  Epoch {epoch}  step {step+1:>3}/{len(train_loader)}  loss={avg:.4f}")
    return total_loss / len(train_loader)


def evaluate_model(model, eval_loader, eval_features, eval_examples):
    """Run inference; eval_loader has only tensor columns (input_ids, attention_mask)."""
    model.eval()
    all_start, all_end = [], []
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            all_start.append(out.start_logits.cpu().numpy())
            all_end.append(out.end_logits.cpu().numpy())

    all_start = np.concatenate(all_start)
    all_end   = np.concatenate(all_end)
    preds, refs = postprocess_qa_predictions(
        eval_examples, eval_features, (all_start, all_end)
    )
    return compute_metrics(preds, refs), preds


# ── 6. Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" NCTB-QA: Bangla Educational QA – POC")
    print("=" * 60)

    # Load dataset
    raw_ds, ds_name = load_bangla_qa()

    train_split = raw_ds.get("train", raw_ds.get("Train"))
    eval_split  = raw_ds.get("validation", raw_ds.get("valid", raw_ds.get("test")))

    # Normalise to SQuAD 2.0 format
    def norm(batch):
        rows = [normalise_squad_example({k: batch[k][i] for k in batch})
                for i in range(len(batch["question"]))]
        return {k: [r[k] for r in rows] for k in rows[0]}

    print("\nNormalising dataset …")
    train_ds = train_split.select(range(min(TRAIN_LIMIT, len(train_split)))).map(
        norm, batched=True, remove_columns=train_split.column_names
    )
    eval_ds = eval_split.select(range(min(EVAL_LIMIT, len(eval_split)))).map(
        norm, batched=True, remove_columns=eval_split.column_names
    )

    print(f"Train examples : {len(train_ds)}")
    print(f"Eval  examples : {len(eval_ds)}")
    print(f"\nSample question: {train_ds[0]['question']}")
    print(f"Sample context : {train_ds[0]['context'][:120]} …")
    print(f"Sample answer  : {train_ds[0]['answers']}")

    # Tokeniser
    print(f"\nLoading tokeniser: {MODEL_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenise
    print("Tokenising train set …")
    train_features = train_ds.map(
        lambda ex: preprocess_train(ex, tokenizer),
        batched=True, remove_columns=train_ds.column_names
    )
    train_features.set_format("torch",
        columns=["input_ids", "attention_mask", "start_positions", "end_positions"])

    print("Tokenising eval set …")
    eval_features = eval_ds.map(
        lambda ex: preprocess_eval(ex, tokenizer),
        batched=True, remove_columns=eval_ds.column_names
    )

    # For the DataLoader only include tensor-compatible columns
    eval_features_loader = eval_features.remove_columns(["offset_mapping", "example_id"])
    eval_features_loader.set_format("torch")

    train_loader = DataLoader(train_features, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=default_data_collator)
    eval_loader  = DataLoader(eval_features_loader, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=default_data_collator)

    # Model
    print(f"\nLoading model: {MODEL_NAME} for QA …")
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)

    total_steps   = len(train_loader) * NUM_EPOCHS
    warmup_steps  = total_steps // 10
    optimizer     = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training
    print(f"\n{'─'*60}")
    print(f"Training for {NUM_EPOCHS} epochs on {ds_name}")
    print(f"{'─'*60}")
    history = []
    for epoch in range(1, NUM_EPOCHS + 1):
        avg_loss = train(model, train_loader, optimizer, scheduler, epoch)
        print(f"→ Epoch {epoch} avg loss: {avg_loss:.4f}")
        history.append(avg_loss)

    # Loss summary table
    print(f"\n{'─'*60}")
    print("Training loss per epoch:")
    df_loss = pd.DataFrame({
        "epoch": list(range(1, NUM_EPOCHS + 1)),
        "avg_loss": [round(l, 4) for l in history],
    })
    print(df_loss.to_string(index=False))
    assert history[-1] < history[0], (
        f"Loss did not decrease: {history[0]:.4f} → {history[-1]:.4f}"
    )
    print(f"✓ Loss decreased: {history[0]:.4f} → {history[-1]:.4f}")

    # Evaluation
    print(f"\n{'─'*60}")
    print("Evaluating on held-out set …")
    metrics, preds = evaluate_model(model, eval_loader, eval_features, eval_ds)
    print(f"\n{'─'*60}")
    print(f"{'Metric':<20} {'Score':>8}")
    print(f"{'─'*28}")
    print(f"{'Exact Match':<20} {metrics['exact_match']:>7.2f}%")
    print(f"{'F1 Score':<20} {metrics['f1']:>7.2f}%")
    print(f"{'─'*28}")

    # Unanswerable stats
    unanswerable_predicted = sum(1 for v in preds.values() if v == "")
    unanswerable_gt = sum(1 for ex in eval_ds if len(ex["answers"]["text"]) == 0)
    print(f"\nUnanswerable questions in eval set : {unanswerable_gt}")
    print(f"Unanswerable questions predicted   : {unanswerable_predicted}")

    # Show example predictions
    print(f"\n{'─'*60}")
    print("Example Predictions:")
    print(f"{'─'*60}")
    shown = 0
    for ex in eval_ds:
        if shown >= 5:
            break
        qid  = ex["id"]
        pred = preds.get(qid, "(not found)")
        gold = ex["answers"]["text"]
        print(f"\n[Q] {ex['question']}")
        print(f"[Gold] {gold if gold else '(unanswerable)'}")
        print(f"[Pred] {pred if pred else '(unanswerable)'}")
        shown += 1

    # Save results
    results = {
        "dataset":          ds_name,
        "model":            MODEL_NAME,
        "train_examples":   len(train_ds),
        "eval_examples":    len(eval_ds),
        "epochs":           NUM_EPOCHS,
        "loss_history":     history,
        "exact_match":      round(metrics["exact_match"], 2),
        "f1":               round(metrics["f1"], 2),
        "unanswerable_gt":  unanswerable_gt,
        "unanswerable_pred":unanswerable_predicted,
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Results saved to results.json")

    # DataFrame summary
    df = pd.DataFrame([results])
    print(f"\nFinal Summary:\n{df[['dataset','model','exact_match','f1']].to_string(index=False)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
