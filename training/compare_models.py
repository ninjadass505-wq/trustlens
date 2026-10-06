"""
TRUSTLENS Phase 10 & 11: Original vs Fine-Tuned Checkpoint Comparison

Evaluates BOTH models on the completely untouched Test Split:
1. Model A: Original Production Baseline Checkpoint
2. Model B: Fine-Tuned Checkpoint (best_val.safetensors)

Generates exact comparative metrics table:
Metric | Original | Fine-Tuned | Difference
"""

import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.evaluate_finetuned import (
    load_model_from_checkpoint,
    evaluate_dataset,
    print_evaluation_summary
)

def run_comparison():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("TRUSTLENS: RIGOROUS HEAD-TO-HEAD COMPARISON ON UNTOUCHED TEST SET")
    print(f"Device: {device}")
    print("=" * 80)

    orig_ckpt = PROJECT_ROOT / "backend" / "models" / "original" / "commfor_model_original.safetensors"
    fine_ckpt = PROJECT_ROOT / "training" / "checkpoints" / "best_val.safetensors"
    test_dir = PROJECT_ROOT / "training" / "data" / "test"

    print(f"Model A (Original Baseline):   {orig_ckpt.name}")
    print(f"Model B (Fine-Tuned Model):    {fine_ckpt.name}")
    print(f"Untouched Test Directory:      {test_dir}\n")

    orig_model = load_model_from_checkpoint(orig_ckpt, device)
    fine_model = load_model_from_checkpoint(fine_ckpt, device)

    # Evaluate on Untouched Test Set
    res_orig = evaluate_dataset(orig_model, test_dir, device)
    res_fine = evaluate_dataset(fine_model, test_dir, device)

    print("\n" + "=" * 80)
    print("PER-IMAGE PREDICTION BREAKDOWN ON UNTOUCHED TEST SET")
    print("=" * 80)
    print(f"{'Image Filename':<28} | {'Truth':<5} | {'Orig Score':<12} | {'Orig Pred':<10} | {'Fine Score':<12} | {'Fine Pred':<10}")
    print("-" * 80)

    for po, pf in zip(res_orig["per_image"], res_fine["per_image"]):
        assert po["file"] == pf["file"]
        orig_stat = "[OK]" if po["correct"] else "[FAIL]"
        fine_stat = "[OK]" if pf["correct"] else "[FAIL]"
        print(f"{po['file']:<28} | {po['truth']:<5} | {po['score_fake_pct']:>9.1f}% AI | {po['pred']:<5} {orig_stat:<6} | {pf['score_fake_pct']:>9.1f}% AI | {pf['pred']:<5} {fine_stat:<6}")
    print("=" * 80)

    # Comparative Metrics Table (Requirement 17)
    metrics_to_compare = [
        ("Overall Accuracy", "accuracy", "%"),
        ("REAL Accuracy", "real_accuracy", "%"),
        ("AI Accuracy (Recall)", "ai_accuracy", "%"),
        ("Precision (AI Class)", "precision", "%"),
        ("F1 Score", "f1", "%"),
        ("False Positive Rate (FPR)", "false_positive_rate", "%"),
        ("False Negative Rate (FNR)", "false_negative_rate", "%"),
        ("ROC-AUC", "roc_auc", "")
    ]

    print("\n" + "=" * 80)
    print("METRIC COMPARISON: ORIGINAL VS FINE-TUNED (UNTOUCHED TEST SET)")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Original Baseline':<18} | {'Fine-Tuned Checkpoint':<22} | {'Difference':<12}")
    print("-" * 80)

    for label, key, unit in metrics_to_compare:
        val_orig = res_orig[key]
        val_fine = res_fine[key]
        diff = val_fine - val_orig
        sign = "+" if diff > 0 else ""
        fmt = ".1f" if unit == "%" else ".4f"
        print(f"{label:<30} | {val_orig:>16{fmt}}{unit} | {val_fine:>20{fmt}}{unit} | {sign}{diff:>{5}{fmt}}{unit}")

    print("=" * 80)

    # Validation Set Check as well
    val_dir = PROJECT_ROOT / "training" / "data" / "val"
    res_orig_val = evaluate_dataset(orig_model, val_dir, device)
    res_fine_val = evaluate_dataset(fine_model, val_dir, device)

    print("\n" + "=" * 80)
    print("VALIDATION SET SUMMARY COMPARISON")
    print("=" * 80)
    print(f"Original Val Accuracy:   {res_orig_val['accuracy']:.1f}% | Real: {res_orig_val['real_accuracy']:.1f}% | AI: {res_orig_val['ai_accuracy']:.1f}% | F1: {res_orig_val['f1']:.1f}%")
    print(f"Fine-Tuned Val Accuracy: {res_fine_val['accuracy']:.1f}% | Real: {res_fine_val['real_accuracy']:.1f}% | AI: {res_fine_val['ai_accuracy']:.1f}% | F1: {res_fine_val['f1']:.1f}%")
    print("=" * 80)

if __name__ == "__main__":
    run_comparison()
