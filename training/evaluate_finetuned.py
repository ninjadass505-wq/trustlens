"""
TRUSTLENS Phase 5 & 16: Evaluation & Baseline Comparator
Evaluates any model checkpoint (Original Baseline or Fine-Tuned)
across Validation and Untouched Test splits with detailed metrics:
- Accuracy, Precision, Recall, F1, AUROC
- Real Accuracy, AI Accuracy, FPR (Real -> AI), FNR (AI -> Real)
- Stratified breakdown across image categories and generators
"""

import os
import sys
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import timm
from safetensors.torch import load_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.detector import IMAGENET_MEAN, IMAGENET_STD

def get_eval_transform():
    return T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=False)
    model.head = nn.Linear(in_features=384, out_features=1, bias=True)

    raw_state = load_file(str(checkpoint_path))
    clean_state = {(k[4:] if k.startswith("vit.") else k): v for k, v in raw_state.items()}
    model.load_state_dict(clean_state, strict=True)
    model.to(device)
    model.eval()
    return model

def compute_roc_auc(labels: List[int], scores: List[float]) -> float:
    """Computes exact Trapezoidal ROC-AUC without external sklearn dependency."""
    pos = [s for l, s in zip(labels, scores) if l == 1]
    neg = [s for l, s in zip(labels, scores) if l == 0]
    if not pos or not neg:
        return 0.5
    # Mann-Whitney U test statistic
    greater = sum(1 for p in pos for n in neg if p > n)
    equal = sum(0.5 for p in pos for n in neg if p == n)
    return (greater + equal) / (len(pos) * len(neg))

def evaluate_dataset(
    model: nn.Module,
    split_dir: Path,
    device: torch.device,
    threshold: float = 0.50
) -> Dict[str, Any]:
    transform = get_eval_transform()

    records = []
    # 1. Real images (Label 0)
    real_files = sorted(list((split_dir / "real").glob("*")))
    for p in real_files:
        records.append({"path": p, "label": 0, "category": "REAL"})

    # 2. AI images (Label 1)
    ai_files = sorted(list((split_dir / "ai").glob("*")))
    for p in ai_files:
        records.append({"path": p, "label": 1, "category": "AI"})

    if not records:
        raise ValueError(f"No evaluation images found in {split_dir}")

    y_true = []
    y_scores = []
    y_preds = []
    per_image_results = []

    with torch.no_grad():
        for r in records:
            img = Image.open(r["path"]).convert("RGB")
            t = transform(img).unsqueeze(0).to(device)
            logit = model(t).item()
            prob = torch.sigmoid(torch.tensor(logit)).item()
            pred = 1 if prob >= threshold else 0

            y_true.append(r["label"])
            y_scores.append(prob)
            y_preds.append(pred)

            per_image_results.append({
                "file": r["path"].name,
                "truth": "REAL" if r["label"] == 0 else "AI",
                "score_fake_pct": round(prob * 100, 1),
                "pred": "AI" if pred == 1 else "REAL",
                "correct": (pred == r["label"])
            })

    # Metric calculations
    tp = sum(1 for yt, yp in zip(y_true, y_preds) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_preds) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_preds) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_preds) if yt == 1 and yp == 0)

    total = len(y_true)
    total_real = sum(1 for yt in y_true if yt == 0)
    total_ai = sum(1 for yt in y_true if yt == 1)

    acc = (tp + tn) / total * 100.0 if total else 0.0
    prec = (tp / (tp + fp)) * 100.0 if (tp + fp) else 0.0
    rec = (tp / (tp + fn)) * 100.0 if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    real_acc = (tn / total_real * 100.0) if total_real else 0.0
    ai_acc = (tp / total_ai * 100.0) if total_ai else 0.0
    fpr = (fp / total_real * 100.0) if total_real else 0.0
    fnr = (fn / total_ai * 100.0) if total_ai else 0.0
    roc_auc = compute_roc_auc(y_true, y_scores)

    return {
        "total": total,
        "total_real": total_real,
        "total_ai": total_ai,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": round(acc, 2),
        "precision": round(prec, 2),
        "recall": round(rec, 2),
        "f1": round(f1, 2),
        "real_accuracy": round(real_acc, 2),
        "ai_accuracy": round(ai_acc, 2),
        "false_positive_rate": round(fpr, 2),
        "false_negative_rate": round(fnr, 2),
        "roc_auc": round(roc_auc, 4),
        "per_image": per_image_results
    }

def print_evaluation_summary(title: str, metrics: Dict[str, Any]):
    print("\n" + "=" * 65)
    print(f"EVALUATION RESULTS: {title}")
    print("=" * 65)
    print(f"Dataset Size:           {metrics['total']} images ({metrics['total_real']} Real, {metrics['total_ai']} AI)")
    print(f"Confusion Matrix:       TP: {metrics['TP']}, TN: {metrics['TN']}, FP: {metrics['FP']}, FN: {metrics['FN']}")
    print("-" * 65)
    print(f"Overall Accuracy:       {metrics['accuracy']:.1f}%")
    print(f"REAL Accuracy:          {metrics['real_accuracy']:.1f}% ({metrics['TN']}/{metrics['total_real']})")
    print(f"AI Accuracy (Recall):   {metrics['ai_accuracy']:.1f}% ({metrics['TP']}/{metrics['total_ai']})")
    print(f"Precision (AI class):   {metrics['precision']:.1f}%")
    print(f"F1 Score:               {metrics['f1']:.1f}%")
    print(f"False Positive Rate:    {metrics['false_positive_rate']:.1f}% (Real -> AI)")
    print(f"False Negative Rate:    {metrics['false_negative_rate']:.1f}% (AI -> Real)")
    print(f"ROC-AUC:                {metrics['roc_auc']:.4f}")
    print("=" * 65)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    orig_ckpt = PROJECT_ROOT / "backend" / "models" / "original" / "commfor_model_original.safetensors"
    val_dir = PROJECT_ROOT / "training" / "data" / "val"
    test_dir = PROJECT_ROOT / "training" / "data" / "test"

    print(f"Running Baseline Evaluation using Original Production Checkpoint: {orig_ckpt}")
    orig_model = load_model_from_checkpoint(orig_ckpt, device)

    val_res = evaluate_dataset(orig_model, val_dir, device)
    print_evaluation_summary("ORIGINAL MODEL ON VALIDATION SET", val_res)

    test_res = evaluate_dataset(orig_model, test_dir, device)
    print_evaluation_summary("ORIGINAL MODEL ON UNTOUCHED TEST SET", test_res)
