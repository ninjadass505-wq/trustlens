# TRUSTLENS V2 — Real-World Image Forensics Benchmark

This directory contains the independent benchmarking infrastructure used to measure the empirical accuracy, false positive rate, and cross-generator generalization of AI image detectors on real-world photographs and synthetic imagery.

---

## 1. Directory Structure

```text
training/benchmark/
├── dataset/
│   ├── manifest.json              # Primary benchmark dataset manifest
│   ├── train/                     # Training split directory (for future model fine-tuning)
│   ├── validation/                # Validation split directory (for threshold calibration)
│   └── test/                      # Held-out test split directory (for final evaluation)
│       ├── real/                  # Genuine camera photographs (DSLR, smartphone, social)
│       └── synthetic/             # Synthetic images (Midjourney, Flux, DALL-E, SDXL, etc.)
├── results/
│   ├── evaluation_raw.json        # Raw per-image inference outputs (logits, patch scores)
│   ├── benchmark_report.json      # Structured summary report & stratified breakdowns
│   ├── benchmark_predictions.csv  # Flat per-image prediction tabular export
│   └── benchmark_summary.csv      # Stratified metrics tabular export
└── scripts/
    ├── manifest_validator.py      # Schema validation & anti-leakage verification tool
    ├── evaluate_current_model.mjs # Multi-scale evaluator for current ONNX production detector
    ├── metrics.py                 # Statistical calculator (Accuracy, F1, AUROC, FPR, FNR)
    ├── run_benchmark.py           # Master end-to-end benchmark orchestrator
    └── test_benchmark.py          # Automated verification test suite
```

---

## 2. Dataset Manifest Specification

The manifest (`training/benchmark/dataset/manifest.json`) is a JSON array of image metadata records.

### Sample Manifest Entry (Real Camera Photo):
```json
{
  "id": "real_cam_001",
  "path": "test/real/real_cam_001.jpg",
  "label": "REAL",
  "source": "RAISE",
  "generator": null,
  "generator_family": null,
  "camera_device": "Nikon D90",
  "resolution": "4288x2848",
  "aspect_ratio": "3:2",
  "compression": "JPEG Q95",
  "scene_category": "outdoor_landscape",
  "split": "test",
  "group_id": "raise_seq_102"
}
```

### Sample Manifest Entry (AI-Generated Image):
```json
{
  "id": "synth_flux1_001",
  "path": "test/synthetic/synth_flux1_001.jpg",
  "label": "SYNTHETIC",
  "source": "Flux1_Dev_Pool",
  "generator": "flux_1",
  "generator_family": "flow_matching",
  "camera_device": null,
  "resolution": "1920x1080",
  "aspect_ratio": "16:9",
  "compression": "JPEG Q88",
  "scene_category": "architecture",
  "split": "test",
  "group_id": "flux_gen_19"
}
```

---

## 3. Strict Anti-Leakage Rules

To prevent benchmarks from becoming artificially easy:
1. **Zero Group Overlap:** `group_id` tracks scene bursts, camera sessions, or prompt seed clusters. No `group_id` may appear in both `train` and `test` splits.
2. **Cross-Generator Separation:** The benchmark partitions generators into **Seen** (in-distribution, e.g. Stable Diffusion 1.4) and **Unseen** (zero-shot generalization, e.g. Flux.1, Midjourney v6, DALL-E 3).
3. **Automated Verification:** The `manifest_validator.py` script automatically verifies and rejects any manifest with split leakage before inference begins.

---

## 4. How to Run the Benchmark

### A. Run Master Benchmark Orchestrator
To validate the manifest, run model inference, compute metrics, and generate reports in one command:
```bash
python training/benchmark/scripts/run_benchmark.py
```

### B. Validate Dataset Manifest Only
To verify manifest schema and check for data leakage:
```bash
python training/benchmark/scripts/manifest_validator.py training/benchmark/dataset/manifest.json
```

### C. Run Inference Only (Node.js ONNX Runtime)
To run the current detector over the manifest using exact TRUSTLENS multi-scale preprocessing:
```bash
node training/benchmark/scripts/evaluate_current_model.mjs training/benchmark/dataset/manifest.json training/benchmark/dataset/ training/benchmark/results/evaluation_raw.json
```

### D. Re-Calculate Metrics from Existing Evaluation
To regenerate metrics and CSVs without re-running inference:
```bash
python training/benchmark/scripts/run_benchmark.py --skip-eval
```

### E. Run the Benchmark Test Suite
To verify label mappings, score orientation, aspect-preserving preprocessing, anti-leakage rules, and determinism:
```bash
python -m unittest training/benchmark/scripts/test_benchmark.py
```

---

## 5. Metrics Produced

1. **Overall Performance (Decided Cases):**
   * **Accuracy:** Overall correctness across decided classifications.
   * **Precision & Recall:** Precision and recall for the synthetic class.
   * **F1 Score:** Harmonic mean of precision and recall.
   * **AUROC:** Area Under the Receiver Operating Characteristic curve across thresholds $\tau \in [0, 100]$.
   * **Inconclusive Rate:** Percentage of images triggering neutral verdicts (`INCONCLUSIVE` / `NEEDS VERIFICATION`).
2. **Critical Risk Metrics:**
   * **REAL IMAGE FALSE POSITIVE RATE (FPR):** Percentage of genuine camera photographs incorrectly classified as synthetic/AI. (Target: $< 5\%$).
   * **SYNTHETIC IMAGE FALSE NEGATIVE RATE (FNR):** Percentage of AI-generated images that slipped past the detector as real.
3. **Confusion Matrix:**
   * $TN$ (Real correctly identified as Real)
   * $FP$ (Real misclassified as AI — False Alarms)
   * $TP$ (AI correctly caught)
   * $FN$ (AI missed as Real)
   * Undecided / Inconclusive counts
4. **Cross-Generator Generalization:**
   * Performance on in-distribution generators (Seen during training).
   * Performance on out-of-distribution generators (Unseen zero-shot models).
5. **Stratified Breakdowns:**
   * By Resolution ($< 1080\text{p}$, $1080\text{p}$, $4\text{K}$)
   * By Aspect Ratio ($16:9$, $4:3$, $3:2$, $1:1$, $9:16$)
   * By Source Device (DSLR, Smartphone, Social Re-encoding)
   * By Compression (Q30, Q50, Q75, Q90, Q95)
   * By Scene Category (Portrait, Landscape, Architecture, Macro, Night)
   * By Generator (Stable Diffusion, Midjourney, Flux, DALL-E, SDXL)
