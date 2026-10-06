"""
TRUSTLENS V2 — Master Benchmark Orchestrator
Coordinates:
1. Manifest validation and zero-leakage verification
2. Model evaluation across multi-scale photographic views
3. Comprehensive statistical metrics and cross-generator calculations
4. Terminal dashboard and CSV/JSON output generation
"""

import argparse
import os
import subprocess
import sys
from manifest_validator import validate_manifest, create_sample_manifest
from metrics import analyze_benchmarks, print_terminal_report

def main():
    parser = argparse.ArgumentParser(description="TRUSTLENS Benchmark Runner")
    parser.add_argument("--manifest", default="training/benchmark/dataset/manifest.json", help="Path to manifest JSON")
    parser.add_argument("--dataset-root", default="training/benchmark/dataset/", help="Root directory of images")
    parser.add_argument("--results-dir", default="training/benchmark/results", help="Directory to save results")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test", "all"], help="Dataset split to evaluate")
    parser.add_argument("--skip-eval", action="store_true", help="Skip model evaluation and re-calculate metrics from raw output")
    parser.add_argument("--init-template", action="store_true", help="Initialize sample template manifest")
    args = parser.parse_args()

    print("====================================================")
    print("TRUSTLENS V2 — BENCHMARK EVALUATION ORCHESTRATOR")
    print("====================================================")

    if args.init_template:
        create_sample_manifest(args.manifest)
        return

    # 1. Validate Manifest
    print(f"\n[Step 1/3] Validating Manifest: {args.manifest}")
    valid, errors, stats = validate_manifest(args.manifest, dataset_root="")
    if not valid:
        print("Manifest validation FAILED:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    print("Manifest validation PASSED (Zero data leakage confirmed).")
    print(f"Total entries: {stats['total_images']} | Splits: {stats['split_counts']}")

    raw_output_path = os.path.join(args.results_dir, "evaluation_raw.json")

    # 2. Run Inference if not skipped
    if not args.skip_eval:
        print(f"\n[Step 2/3] Running Model Evaluation via Node.js ONNX Runtime...")
        eval_script = os.path.join(os.path.dirname(__file__), "evaluate_current_model.mjs")
        cmd = ["node", eval_script, args.manifest, args.dataset_root, raw_output_path]
        try:
            res = subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Model evaluation failed with exit code {e.returncode}")
            sys.exit(e.returncode)
        except FileNotFoundError:
            print("Error: Node.js is required to execute evaluate_current_model.mjs.")
            sys.exit(1)
    else:
        print(f"\n[Step 2/3] Skipping model evaluation (--skip-eval specified).")

    # 3. Calculate Metrics & Breakdowns
    print(f"\n[Step 3/3] Computing Statistical Metrics & Cross-Generator Breakdowns...")
    if not os.path.exists(raw_output_path):
        print(f"Error: Raw evaluation file not found at {raw_output_path}")
        sys.exit(1)

    report = analyze_benchmarks(raw_output_path, args.results_dir)
    print_terminal_report(report)

    print("====================================================")
    print(f"Benchmark results successfully exported to:")
    print(f"  - Report (JSON):      {os.path.join(args.results_dir, 'benchmark_report.json')}")
    print(f"  - Predictions (CSV):  {os.path.join(args.results_dir, 'benchmark_predictions.csv')}")
    print(f"  - Summary (CSV):      {os.path.join(args.results_dir, 'benchmark_summary.csv')}")
    print("====================================================")

if __name__ == "__main__":
    main()
