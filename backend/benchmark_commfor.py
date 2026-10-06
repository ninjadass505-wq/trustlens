"""
TRUSTLENS V2 — Community Forensics 224 Benchmark & Calibration Suite

Compares:
1. OLD Detector: capcheck/ai-image-detection (ONNX Q4 CIFAKE)
2. NEW Detector: OwensLab/commfor-model-224 (Community Forensics ViT-S/16)

Evaluates BOTH:
A. Single Global View
B. Multi-Scale 5-Patch Forensic Pipeline (Global + Center + Native 1:1 Crops)
"""

import os
import io
import time
import math
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.detector import detector, IMAGENET_MEAN, IMAGENET_STD

# ----------------------------------------------------------------------
# 1. OLD MODEL WRAPPER (capcheck/ai-image-detection ONNX)
# ----------------------------------------------------------------------
class OldCifakeOnnxDetector:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, pil_img: Image.Image) -> np.ndarray:
        # CIFAKE preprocessing: Resize to 224x224, rescale 1/255, normalize mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        img = pil_img.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - np.array([0.5, 0.5, 0.5], dtype=np.float32)) / np.array([0.5, 0.5, 0.5], dtype=np.float32)
        arr = np.transpose(arr, (2, 0, 1)) # [3, 224, 224]
        return np.expand_dims(arr, axis=0) # [1, 3, 224, 224]

    def predict_single(self, pil_img: Image.Image) -> Tuple[float, float, float]:
        t0 = time.perf_counter()
        inp = self.preprocess(pil_img)
        outputs = self.session.run(None, {self.input_name: inp})
        logits = outputs[0][0]
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        prob_real = float(probs[0])
        prob_fake = float(probs[1])
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return prob_fake * 100.0, prob_real * 100.0, latency_ms

# ----------------------------------------------------------------------
# 2. MULTI-SCALE PATCH GENERATOR (Replicates TRUSTLENS forensics.js)
# ----------------------------------------------------------------------
def generate_forensic_patches_py(img: Image.Image, patch_size: int = 224) -> List[Dict[str, Any]]:
    w, h = img.size
    patches = []

    # 1. Global View (Aspect preserved, letterboxed or resized)
    global_img = img.convert('RGB').resize((patch_size, patch_size), Image.Resampling.BILINEAR)
    patches.append({
        "id": "global",
        "name": "Global Context View",
        "type": "global",
        "image": global_img
    })

    # 2. Native 1:1 Center Patch
    if w >= patch_size and h >= patch_size:
        cx = (w - patch_size) // 2
        cy = (h - patch_size) // 2
        center_img = img.convert('RGB').crop((cx, cy, cx + patch_size, cy + patch_size))
        patches.append({
            "id": "center",
            "name": "Native Center Patch",
            "type": "native_center",
            "image": center_img
        })

        # 3. Top-Left Patch
        tl_img = img.convert('RGB').crop((0, 0, patch_size, patch_size))
        patches.append({
            "id": "top_left",
            "name": "Native Top-Left Patch",
            "type": "native_corner",
            "image": tl_img
        })

        # 4. Bottom-Right Patch
        br_img = img.convert('RGB').crop((w - patch_size, h - patch_size, w, h))
        patches.append({
            "id": "bottom_right",
            "name": "Native Bottom-Right Patch",
            "type": "native_corner",
            "image": br_img
        })

        # 5. High-Frequency Texture Crop (Top-Right or Bottom-Left)
        tr_img = img.convert('RGB').crop((w - patch_size, 0, w, patch_size))
        patches.append({
            "id": "top_right",
            "name": "Native Top-Right Patch",
            "type": "native_corner",
            "image": tr_img
        })
    else:
        # For images smaller than 224x224, upsample global view
        upsampled = img.convert('RGB').resize((patch_size, patch_size), Image.Resampling.BILINEAR)
        patches.append({
            "id": "center_upsampled",
            "name": "Upsampled Center",
            "type": "upsampled",
            "image": upsampled
        })

    return patches

# ----------------------------------------------------------------------
# 3. EVIDENCE AGGREGATION (0.7 * Trimmed Mean + 0.3 * Median)
# ----------------------------------------------------------------------
def aggregate_patch_evidence_py(patch_scores: List[float]) -> Dict[str, Any]:
    scores = sorted(patch_scores)
    n = len(scores)

    # Median
    if n % 2 == 1:
        median = scores[n // 2]
    else:
        median = (scores[n // 2 - 1] + scores[n // 2]) / 2.0

    # Trimmed Mean: Trim highest and lowest if 4 or more patches
    if n >= 4:
        trimmed = scores[1:-1]
        mean_trimmed = float(np.mean(trimmed))
    else:
        mean_trimmed = float(np.mean(scores))

    # Existing TRUSTLENS consensus formula: 0.7 * trimmed_mean + 0.3 * median
    final_synthetic = 0.7 * mean_trimmed + 0.3 * median
    final_real = 100.0 - final_synthetic

    score_gap = abs(final_synthetic - final_real)
    score_range = max(scores) - min(scores)

    # Calibrated decision rules:
    # Strong consensus (>65% synth with >25% gap) -> LIKELY AI-GENERATED
    # Strong authentic (<35% synth with >25% gap) -> LIKELY REAL
    # Small gap or high patch contradiction (>55% range) -> INCONCLUSIVE
    if score_range > 55.0 or score_gap < 15.0:
        label = "INCONCLUSIVE"
    elif final_synthetic >= 60.0:
        label = "LIKELY AI-GENERATED"
    elif final_synthetic <= 40.0:
        label = "LIKELY REAL"
    else:
        label = "INCONCLUSIVE"

    return {
        "synthetic_score": round(final_synthetic, 1),
        "real_score": round(final_real, 1),
        "score_gap": round(score_gap, 1),
        "disagreement_range": round(score_range, 1),
        "label": label
    }

# ----------------------------------------------------------------------
# 4. BENCHMARK EXECUTION
# ----------------------------------------------------------------------
def run_benchmark():
    # 1. Setup Models
    old_model_path = os.path.abspath("frontend/models/onnx-community/ai-image-detection-ONNX/onnx/model_q4.onnx")
    old_detector = OldCifakeOnnxDetector(old_model_path)

    if not detector.loaded:
        detector.load_model()

    test_images_dir = Path("C:/Users/keert/.gemini/antigravity-ide/brain/859c1a87-6841-4cfc-9230-09667474976c/scratch/test_images")
    
    # Dataset items
    dataset = [
        # REAL PHOTOGRAPHS
        {"name": "real_fruits.jpg", "category": "Objects/Tabletop", "truth": "REAL"},
        {"name": "real_messi.jpg", "category": "Outdoor Sports/Action", "truth": "REAL"},
        {"name": "real_building.jpg", "category": "Architecture/Facade", "truth": "REAL"},
        {"name": "real_baboon.jpg", "category": "Wildlife/Hair Texture", "truth": "REAL"},
        {"name": "real_grace_hopper.jpg", "category": "Portrait/Person", "truth": "REAL"},
        # AI GENERATED IMAGES
        {"name": "ai_dalle2_01.png", "category": "DALL-E 2 / Generative Art", "truth": "AI"},
        {"name": "ai_dalle2_02.png", "category": "DALL-E 2 / Portrait", "truth": "AI"},
        {"name": "ai_dalle2_03.png", "category": "DALL-E 2 / Photorealistic Scene", "truth": "AI"},
        {"name": "ai_dalle2_04.png", "category": "DALL-E 2 / Landscape (Smooth Center)", "truth": "AI"},
        {"name": "ai_dalle2_05.png", "category": "DALL-E 2 / Complex Composition", "truth": "AI"},
    ]

    print("=" * 85)
    print("TRUSTLENS V2 FORENSIC BENCHMARK: CIFAKE vs COMMUNITY FORENSICS 224")
    print("=" * 85)
    print(f"{'Image':<22} | {'Truth':<5} | {'Old Global':<12} | {'New Global':<12} | {'New Multi-Scale':<16} | {'New Decision':<12}")
    print("-" * 85)

    records = []

    for item in dataset:
        img_path = test_images_dir / item["name"]
        if not img_path.exists():
            continue

        img = Image.open(img_path)

        # 1. Old Model Single View
        old_synth, old_real, old_lat = old_detector.predict_single(img)
        old_dec = "AI" if old_synth > 50 else "REAL"

        # 2. New Model Single View (Global resize)
        single_patch = [{"id": "global", "image": img.resize((224, 224))}]
        new_single_res = detector.analyze_patches(single_patch)
        new_single_synth = new_single_res["results"][0]["syntheticScore"]
        new_single_dec = "AI" if new_single_synth > 50 else "REAL"

        # 3. New Model Multi-Scale 5-Patch Pipeline
        patches = generate_forensic_patches_py(img, 224)
        new_multi_res = detector.analyze_patches(patches)
        patch_synth_scores = [r["syntheticScore"] for r in new_multi_res["results"]]
        multi_agg = aggregate_patch_evidence_py(patch_synth_scores)

        records.append({
            "name": item["name"],
            "truth": item["truth"],
            "category": item["category"],
            "dimensions": f"{img.width}x{img.height}",
            "old_global_synth": old_synth,
            "old_global_dec": old_dec,
            "new_global_synth": new_single_synth,
            "new_global_dec": new_single_dec,
            "new_multi_synth": multi_agg["synthetic_score"],
            "new_multi_label": multi_agg["label"],
            "multi_latency_ms": new_multi_res["inference"]["totalLatencyMs"]
        })

        print(f"{item['name']:<22} | {item['truth']:<5} | {old_synth:>9.1f}% AI | {new_single_synth:>9.1f}% AI | {multi_agg['synthetic_score']:>11.1f}% AI | {multi_agg['label']:<12}")

    print("=" * 85)

    # Metric calculations
    def compute_metrics(recs, key_synth, threshold=50.0):
        tp = sum(1 for r in recs if r["truth"] == "AI" and r[key_synth] >= threshold)
        fn = sum(1 for r in recs if r["truth"] == "AI" and r[key_synth] < threshold)
        tn = sum(1 for r in recs if r["truth"] == "REAL" and r[key_synth] < threshold)
        fp = sum(1 for r in recs if r["truth"] == "REAL" and r[key_synth] >= threshold)

        total_real = sum(1 for r in recs if r["truth"] == "REAL")
        total_ai = sum(1 for r in recs if r["truth"] == "AI")

        accuracy = (tp + tn) / len(recs) * 100.0 if recs else 0
        precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        fpr = (fp / total_real * 100.0) if total_real > 0 else 0
        fnr = (fn / total_ai * 100.0) if total_ai > 0 else 0

        return {
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Accuracy": accuracy, "Precision": precision,
            "Recall": recall, "F1": f1, "FPR": fpr, "FNR": fnr
        }

    m_old_global = compute_metrics(records, "old_global_synth")
    m_new_global = compute_metrics(records, "new_global_synth")
    m_new_multi = compute_metrics(records, "new_multi_synth")

    print("\n" + "=" * 75)
    print("DETAILED PERFORMANCE COMPARISON ACROSS EVALUATION MODES")
    print("=" * 75)
    print(f"{'Metric':<28} | {'Old CIFAKE (Global)':<18} | {'CommFor (Global)':<16} | {'CommFor (Multi-Scale)'}")
    print("-" * 75)
    print(f"{'True Positives (AI detected)':<28} | {m_old_global['TP']:>18} | {m_new_global['TP']:>16} | {m_new_multi['TP']:>20}")
    print(f"{'True Negatives (Real detected)':<28} | {m_old_global['TN']:>18} | {m_new_global['TN']:>16} | {m_new_multi['TN']:>20}")
    print(f"{'False Positives (Real -> AI)':<28} | {m_old_global['FP']:>18} | {m_new_global['FP']:>16} | {m_new_multi['FP']:>20}")
    print(f"{'False Negatives (AI -> Real)':<28} | {m_old_global['FN']:>18} | {m_new_global['FN']:>16} | {m_new_multi['FN']:>20}")
    print(f"{'False Positive Rate (FPR)':<28} | {m_old_global['FPR']:>17.1f}% | {m_new_global['FPR']:>15.1f}% | {m_new_multi['FPR']:>19.1f}%")
    print(f"{'False Negative Rate (FNR)':<28} | {m_old_global['FNR']:>17.1f}% | {m_new_global['FNR']:>15.1f}% | {m_new_multi['FNR']:>19.1f}%")
    print(f"{'Accuracy':<28} | {m_old_global['Accuracy']:>17.1f}% | {m_new_global['Accuracy']:>15.1f}% | {m_new_multi['Accuracy']:>19.1f}%")
    print(f"{'Precision':<28} | {m_old_global['Precision']:>17.1f}% | {m_new_global['Precision']:>15.1f}% | {m_new_multi['Precision']:>19.1f}%")
    print(f"{'Recall':<28} | {m_old_global['Recall']:>17.1f}% | {m_new_global['Recall']:>15.1f}% | {m_new_multi['Recall']:>19.1f}%")
    print(f"{'F1 Score':<28} | {m_old_global['F1']:>17.1f}% | {m_new_global['F1']:>15.1f}% | {m_new_multi['F1']:>19.1f}%")
    print("=" * 75)

    # Multi-scale consensus analysis
    inconclusive_count = sum(1 for r in records if r["new_multi_label"] == "INCONCLUSIVE")
    print(f"\nMulti-Scale Forensic Consensus Safety:")
    print(f"  Inconclusive Rate: {inconclusive_count}/{len(records)} ({inconclusive_count/len(records)*100:.1f}%)")
    print(f"  Avg Multi-Scale 5-Patch Latency: {np.mean([r['multi_latency_ms'] for r in records]):.1f} ms\n")

if __name__ == "__main__":
    run_benchmark()
