"""
TRUSTLENS Benchmark Statistical Metrics & Cross-Generator Reporting Engine
Computes:
- Classification accuracy, precision, recall, F1, AUROC
- False Positive Rate (FPR) on REAL images (Critical Metric)
- False Negative Rate (FNR) on SYNTHETIC images
- Confusion Matrix with Inconclusive tracking
- Stratified breakdowns: Resolution, Aspect Ratio, Source, Compression, Scene, Generator
- Cross-Generator Evaluation Mode (Seen vs Unseen Generative Families)
- Exports formatted JSON and CSV summaries
"""

import json
import csv
import math
import os
import sys
from typing import Dict, List, Any, Tuple

# Default known training generators for CIFAKE model baseline
SEEN_GENERATORS_DEFAULT = {"stable_diffusion_1.4", "cifake", "stable_diffusion_1.5"}

def calculate_auroc(y_true: List[int], scores: List[float]) -> float:
    """
    Computes Area Under the Receiver Operating Characteristic Curve (AUROC)
    using the trapezoidal rule over sorted thresholds.
    y_true: 1 for SYNTHETIC, 0 for REAL
    scores: synthetic risk scores (0 to 100)
    """
    if len(set(y_true)) < 2:
        return float("nan") # Requires both positive and negative samples

    # Sort descending by score
    combined = sorted(zip(scores, y_true), key=lambda x: x[0], reverse=True)
    
    pos_count = sum(y_true)
    neg_count = len(y_true) - pos_count

    if pos_count == 0 or neg_count == 0:
        return float("nan")

    tpr_list = [0.0]
    fpr_list = [0.0]

    tp = 0
    fp = 0

    prev_score = None
    for score, label in combined:
        if prev_score is not None and score != prev_score:
            tpr_list.append(tp / pos_count)
            fpr_list.append(fp / neg_count)

        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score

    tpr_list.append(1.0)
    fpr_list.append(1.0)

    # Numerical trapezoid integration
    area = 0.0
    for i in range(1, len(fpr_list)):
        width = fpr_list[i] - fpr_list[i - 1]
        height = (tpr_list[i] + tpr_list[i - 1]) / 2.0
        area += width * height

    return round(area, 4)

def evaluate_subset(items: List[Dict[str, Any]], name: str = "Subset") -> Dict[str, Any]:
    """Computes comprehensive classification metrics on a subset of items."""
    total = len(items)
    if total == 0:
        return {"name": name, "total": 0}

    y_true = []
    scores = []

    tp = 0 # Synth classified as Synth
    fp = 0 # Real classified as Synth
    tn = 0 # Real classified as Real
    fn = 0 # Synth classified as Real
    inconclusive = 0

    real_total = 0
    synth_total = 0

    for item in items:
        gt = item.get("ground_truth", "").upper()
        inference = item.get("inference", {})
        decision = inference.get("final_decision", "INCONCLUSIVE")
        synth_score = float(inference.get("synthetic_score", 50))

        is_synth = 1 if gt == "SYNTHETIC" else 0
        y_true.append(is_synth)
        scores.append(synth_score)

        if gt == "REAL":
            real_total += 1
        else:
            synth_total += 1

        if decision == "INCONCLUSIVE" or decision == "NEEDS VERIFICATION" or decision == "OUTSIDE MODEL SCOPE":
            inconclusive += 1
        elif decision == "LIKELY AI-GENERATED":
            if is_synth == 1:
                tp += 1
            else:
                fp += 1
        elif decision == "LIKELY REAL":
            if is_synth == 0:
                tn += 1
            else:
                fn += 1

    decided = tp + tn + fp + fn
    accuracy = (tp + tn) / decided * 100.0 if decided > 0 else 0.0
    precision = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Critical False Positive Rate on Real Images
    fpr_real = (fp / real_total * 100.0) if real_total > 0 else 0.0
    # False Negative Rate on Synthetic Images
    fnr_synth = (fn / synth_total * 100.0) if synth_total > 0 else 0.0

    auroc = calculate_auroc(y_true, scores)

    return {
        "name": name,
        "total_samples": total,
        "real_samples": real_total,
        "synthetic_samples": synth_total,
        "decided_samples": decided,
        "inconclusive_samples": inconclusive,
        "inconclusive_rate": round(inconclusive / total * 100.0, 1),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1, 2),
        "fpr_real": round(fpr_real, 2),
        "fnr_synthetic": round(fnr_synth, 2),
        "auroc": auroc,
        "confusion_matrix": {
            "TN_real_correct": tn,
            "FP_real_as_ai": fp,
            "FN_ai_as_real": fn,
            "TP_ai_correct": tp,
            "undecided_inconclusive": inconclusive
        }
    }

def analyze_benchmarks(raw_eval_path: str, results_dir: str, seen_generators: set = None) -> Dict[str, Any]:
    """Generates the full benchmark report, stratified breakdowns, and CSVs."""
    if seen_generators is None:
        seen_generators = SEEN_GENERATORS_DEFAULT

    with open(raw_eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("Warning: evaluation_raw.json is empty.")
        return {}

    os.makedirs(results_dir, exist_ok=True)

    # 1. Overall Metrics
    overall = evaluate_subset(data, name="Overall Test Set")

    # 2. Cross-Generator Evaluation Mode (Seen vs Unseen)
    seen_items = []
    unseen_items = []
    for item in data:
        if item.get("ground_truth") == "SYNTHETIC":
            gen = item.get("generator", "")
            if gen in seen_generators:
                seen_items.append(item)
            else:
                unseen_items.append(item)
        else:
            # For real images, include them in both to measure comparative discrimination
            seen_items.append(item)
            unseen_items.append(item)

    seen_metrics = evaluate_subset(seen_items, name="Cross-Gen: Seen Generators (CIFAKE/SD 1.4)")
    unseen_metrics = evaluate_subset(unseen_items, name="Cross-Gen: Unseen Zero-Shot Generators (Midjourney, Flux, DALL-E, SDXL)")

    # 3. Stratified Breakdowns
    def breakdown_by(key: str) -> Dict[str, Dict]:
        groups = {}
        for item in data:
            val = str(item.get(key) or "unspecified")
            if val not in groups:
                groups[val] = []
            groups[val].append(item)
        return {val: evaluate_subset(items, name=f"{key}:{val}") for val, items in sorted(groups.items())}

    breakdowns = {
        "by_resolution": breakdown_by("resolution"),
        "by_aspect_ratio": breakdown_by("aspect_ratio"),
        "by_source": breakdown_by("source"),
        "by_compression": breakdown_by("compression"),
        "by_scene_category": breakdown_by("scene_category"),
        "by_generator": breakdown_by("generator")
    }

    report = {
        "model_name": "capcheck/ai-image-detection (ONNX q4 baseline)",
        "model_checkpoint": "frontend/models/onnx-community/ai-image-detection-ONNX/onnx/model_q4.onnx",
        "dataset_evaluated": raw_eval_path,
        "overall": overall,
        "cross_generator": {
            "seen_generators": seen_metrics,
            "unseen_generators": unseen_metrics
        },
        "breakdowns": breakdowns
    }

    # 4. Save JSON Report
    json_path = os.path.join(results_dir, "benchmark_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # 5. Save Predictions CSV
    csv_preds_path = os.path.join(results_dir, "benchmark_predictions.csv")
    with open(csv_preds_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "path", "ground_truth", "source", "generator", "resolution", "aspect_ratio",
            "compression", "scene_category", "real_score", "synthetic_score",
            "evidence_consistency", "final_decision", "patch_std_dev", "disagreement_range"
        ])
        for item in data:
            inf = item.get("inference", {})
            writer.writerow([
                item.get("id"),
                item.get("path"),
                item.get("ground_truth"),
                item.get("source"),
                item.get("generator"),
                item.get("resolution"),
                item.get("aspect_ratio"),
                item.get("compression"),
                item.get("scene_category"),
                inf.get("real_score"),
                inf.get("synthetic_score"),
                inf.get("evidence_consistency"),
                inf.get("final_decision"),
                inf.get("patch_std_dev"),
                inf.get("disagreement_range")
            ])

    # 6. Save Summary CSV
    csv_summary_path = os.path.join(results_dir, "benchmark_summary.csv")
    with open(csv_summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Group", "Total", "Accuracy%", "Precision%", "Recall%", "F1%", "AUROC", "FPR_Real%", "FNR_Synthetic%", "Inconclusive%"])
        
        o = overall
        writer.writerow(["Overall", "All Test", o["total_samples"], o["accuracy"], o["precision"], o["recall"], o["f1_score"], o["auroc"], o["fpr_real"], o["fnr_synthetic"], o["inconclusive_rate"]])

        sg = seen_metrics
        writer.writerow(["CrossGen", "Seen Generators", sg["total_samples"], sg["accuracy"], sg["precision"], sg["recall"], sg["f1_score"], sg["auroc"], sg["fpr_real"], sg["fnr_synthetic"], sg["inconclusive_rate"]])

        ug = unseen_metrics
        writer.writerow(["CrossGen", "Unseen Generators", ug["total_samples"], ug["accuracy"], ug["precision"], ug["recall"], ug["f1_score"], ug["auroc"], ug["fpr_real"], ug["fnr_synthetic"], ug["inconclusive_rate"]])

        for b_name, b_data in breakdowns.items():
            for g_name, g_metrics in b_data.items():
                writer.writerow([b_name, g_name, g_metrics["total_samples"], g_metrics.get("accuracy", 0), g_metrics.get("precision", 0), g_metrics.get("recall", 0), g_metrics.get("f1_score", 0), g_metrics.get("auroc", 0), g_metrics.get("fpr_real", 0), g_metrics.get("fnr_synthetic", 0), g_metrics.get("inconclusive_rate", 0)])

    return report

def print_terminal_report(report: Dict[str, Any]):
    """Prints a clear terminal dashboard matching Section 8 format."""
    if not report:
        return

    o = report["overall"]
    cm = o.get("confusion_matrix", {})
    cg = report["cross_generator"]

    print("\n" + "=" * 70)
    print("TRUSTLENS IMAGE FORENSICS BENCHMARK REPORT")
    print(f"MODEL: {report['model_name']}")
    print("=" * 70)
    print("\nOVERALL METRICS (Decided Cases):")
    print(f"  Total Images:                    {o['total_samples']}")
    print(f"  Real Photographs:                {o['real_samples']}")
    print(f"  Synthetic Images:                {o['synthetic_samples']}")
    print(f"  Decided Assessments:             {o['decided_samples']}")
    print(f"  Inconclusive Assessments:        {o['inconclusive_samples']} ({o['inconclusive_rate']}%)")
    print("-" * 70)
    print(f"  Accuracy:                        {o['accuracy']}%")
    print(f"  Precision (Synthetic Class):     {o['precision']}%")
    print(f"  Recall (Synthetic Class):        {o['recall']}%")
    print(f"  F1 Score:                        {o['f1_score']}")
    print(f"  AUROC (Area Under ROC Curve):    {o['auroc']}")
    print("-" * 70)
    print(f"  REAL IMAGE FALSE POSITIVE RATE:  {o['fpr_real']}%")
    print(f"  SYNTHETIC IMAGE FALSE NEGATIVE:  {o['fnr_synthetic']}%")
    print("-" * 70)
    print("CONFUSION MATRIX:")
    print(f"  True Real (Correct Camera):      {cm.get('TN_real_correct', 0)}")
    print(f"  False Positive (Real Flagged AI):{cm.get('FP_real_as_ai', 0)}  <-- Critical Failure Metric")
    print(f"  True Synthetic (Caught AI):      {cm.get('TP_ai_correct', 0)}")
    print(f"  False Negative (AI Missed Real): {cm.get('FN_ai_as_real', 0)}")
    print(f"  Inconclusive / Out of Scope:     {cm.get('undecided_inconclusive', 0)}")

    print("\n" + "=" * 70)
    print("CROSS-GENERATOR GENERALIZATION RESULTS:")
    print("=" * 70)
    seen = cg["seen_generators"]
    unseen = cg["unseen_generators"]
    print(f"  Seen Generators (In-Distribution):")
    print(f"    Total: {seen['total_samples']} | Acc: {seen.get('accuracy',0)}% | F1: {seen.get('f1_score',0)} | FPR on Real: {seen.get('fpr_real',0)}%")
    print(f"  Unseen Generators (Zero-Shot Generalization):")
    print(f"    Total: {unseen['total_samples']} | Acc: {unseen.get('accuracy',0)}% | F1: {unseen.get('f1_score',0)} | FPR on Real: {unseen.get('fpr_real',0)}%")

    print("\n" + "=" * 70)
    print("BREAKDOWN BY GENERATOR:")
    print("-" * 70)
    gen_b = report["breakdowns"]["by_generator"]
    for gen, metrics in gen_b.items():
        if gen != "None":
            print(f"  {gen:<24} | Total: {metrics['total_samples']:<3} | Acc: {metrics.get('accuracy',0):>5}% | FNR (Missed): {metrics.get('fnr_synthetic',0):>5}% | Inconclusive: {metrics.get('inconclusive_samples',0):>3}")

    print("\n" + "=" * 70)
    print("BREAKDOWN BY ASPECT RATIO & COMPRESSION:")
    print("-" * 70)
    for ar, m in report["breakdowns"]["by_aspect_ratio"].items():
        print(f"  Aspect {ar:<16} | Total: {m['total_samples']:<3} | Acc: {m.get('accuracy',0):>5}% | FPR on Real: {m.get('fpr_real',0):>5}%")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    raw_path = sys.argv[1] if len(sys.argv) > 1 else "training/benchmark/results/evaluation_raw.json"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "training/benchmark/results"
    
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} does not exist. Run evaluation first.")
        sys.exit(1)

    rep = analyze_benchmarks(raw_path, out_dir)
    print_terminal_report(rep)
