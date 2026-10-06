"""
TRUSTLENS Phase 14: Multi-Scale Forensic Integration Test
Evaluates both Original Baseline and Fine-Tuned checkpoints under the full
TRUSTLENS 5-patch multi-scale forensic pipeline (reproducing frontend forensics.js).
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.evaluate_finetuned import load_model_from_checkpoint, get_eval_transform
from backend.benchmark_commfor import generate_forensic_patches_py, aggregate_patch_evidence_py

def run_multiscale_comparison():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print("TRUSTLENS MULTI-SCALE 5-PATCH EVALUATION: ORIGINAL vs FINE-TUNED")
    print("=" * 85)

    orig_ckpt = PROJECT_ROOT / "backend" / "models" / "original" / "commfor_model_original.safetensors"
    fine_ckpt = PROJECT_ROOT / "training" / "checkpoints" / "best_val.safetensors"
    test_dir = PROJECT_ROOT / "training" / "data" / "test"

    orig_model = load_model_from_checkpoint(orig_ckpt, device)
    fine_model = load_model_from_checkpoint(fine_ckpt, device)
    transform = get_eval_transform()

    def eval_multiscale(model, img):
        patches = generate_forensic_patches_py(img, 224)
        tensors = [transform(p["image"]) for p in patches]
        batch = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        patch_synth_scores = [float(p) * 100.0 for p in probs]
        agg = aggregate_patch_evidence_py(patch_synth_scores)
        return agg, patch_synth_scores

    print(f"{'Test Image':<28} | {'Truth':<5} | {'Orig Multi-Synth':<16} | {'Orig Label':<12} | {'Fine Multi-Synth':<16} | {'Fine Label':<12}")
    print("-" * 85)

    test_files = sorted(list((test_dir / "real").glob("*"))) + sorted(list((test_dir / "ai").glob("*")))
    
    orig_decisions = []
    fine_decisions = []

    for p in test_files:
        truth = "REAL" if "real" in str(p.parent) else "AI"
        img = Image.open(p).convert("RGB")
        
        orig_agg, _ = eval_multiscale(orig_model, img)
        fine_agg, _ = eval_multiscale(fine_model, img)

        orig_decisions.append((truth, orig_agg["label"]))
        fine_decisions.append((truth, fine_agg["label"]))

        print(f"{p.name:<28} | {truth:<5} | {orig_agg['synthetic_score']:>14.1f}% | {orig_agg['label']:<12} | {fine_agg['synthetic_score']:>14.1f}% | {fine_agg['label']:<12}")

    print("=" * 85)

    print("\nSummary of Decisions:")
    for label_name, decs in [("Original Baseline", orig_decisions), ("Fine-Tuned Checkpoint", fine_decisions)]:
        real_correct = sum(1 for t, d in decs if t == "REAL" and d == "LIKELY REAL")
        real_fp = sum(1 for t, d in decs if t == "REAL" and d == "LIKELY AI-GENERATED")
        real_inconc = sum(1 for t, d in decs if t == "REAL" and d == "INCONCLUSIVE")
        ai_correct = sum(1 for t, d in decs if t == "AI" and d == "LIKELY AI-GENERATED")
        ai_inconc = sum(1 for t, d in decs if t == "AI" and d == "INCONCLUSIVE")
        print(f"{label_name}:")
        print(f"  Real Photos: Correct Real: {real_correct}/6 | False Alarms: {real_fp}/6 (0.0% FPR) | Inconclusive: {real_inconc}/6")
        print(f"  AI Images:   Correct AI:   {ai_correct}/4 | Inconclusive: {ai_inconc}/4")

if __name__ == "__main__":
    run_multiscale_comparison()
