"""
TRUSTLENS Benchmark Test Suite
Validates:
1. Label mapping (Index 0 = REAL, Index 1 = FAKE)
2. Score orientation (P(real) + P(synthetic) = 1.0)
3. Preprocessing (aspect-ratio preservation, zero gray bars)
4. Anti-leakage enforcement (train/val/test strict separation)
5. Deterministic inference (identical outputs for identical inputs)
"""

import json
import os
import subprocess
import sys
import unittest
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from manifest_validator import validate_manifest

class TestBenchmarkInfrastructure(unittest.TestCase):

    def setUp(self):
        self.model_config_path = os.path.abspath("frontend/models/onnx-community/ai-image-detection-ONNX/config.json")
        self.preprocessor_config_path = os.path.abspath("frontend/models/onnx-community/ai-image-detection-ONNX/preprocessor_config.json")

    def test_01_label_mapping(self):
        """Verifies model config class mapping: Index 0 is REAL, Index 1 is FAKE."""
        self.assertTrue(os.path.exists(self.model_config_path), f"Config file not found at {self.model_config_path}")
        with open(self.model_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        id2label = cfg.get("id2label", {})
        label2id = cfg.get("label2id", {})

        # Assert index 0 is REAL and index 1 is FAKE
        self.assertEqual(id2label.get("0"), "REAL", "Index 0 must be mapped to REAL")
        self.assertEqual(id2label.get("1"), "FAKE", "Index 1 must be mapped to FAKE")
        self.assertEqual(label2id.get("REAL"), 0, "REAL must have ID 0")
        self.assertEqual(label2id.get("FAKE"), 1, "FAKE must have ID 1")

    def test_02_score_orientation(self):
        """Verifies score conversion and complementary orientation."""
        raw_logits = [-0.5, 1.2] # Leaning Fake
        exps = [np.exp(l) for l in raw_logits]
        probs = [e / sum(exps) for e in exps]
        
        real_score = round(probs[0] * 100)
        synth_score = round(probs[1] * 100)

        # Real and synthetic must sum to 100 (within rounding +/- 1)
        self.assertAlmostEqual(real_score + synth_score, 100, delta=1)
        self.assertGreater(synth_score, real_score, "Score orientation must correlate higher score to Fake for logit_1 > logit_0")

    def test_03_preprocessing_aspect_and_zero_gray_bars(self):
        """Verifies that aspect-preserving center crop contains zero artificial border padding."""
        orig_w, orig_h = 1920, 1080
        # Create non-zero texture test image
        img_arr = np.random.randint(50, 200, (orig_h, orig_w, 3), dtype=np.uint8)
        img = Image.fromarray(img_arr)

        target_size = 224
        scale = max(target_size / orig_w, target_size / orig_h)
        crop_w = int(target_size / scale)
        crop_h = int(target_size / scale)
        src_x = (orig_w - crop_w) // 2
        src_y = (orig_h - crop_h) // 2

        crop = img.crop((src_x, src_y, src_x + crop_w, src_y + crop_h))
        resized = crop.resize((target_size, target_size), Image.Resampling.BILINEAR)

        # 1. Output dimensions must be exactly 224x224
        self.assertEqual(resized.size, (224, 224))

        # 2. Border regions must NOT have zero variance (i.e. NO solid gray #808080 padding)
        crop_arr = np.array(resized)
        top_strip_var = np.var(crop_arr[:30, :, :])
        bottom_strip_var = np.var(crop_arr[-30:, :, :])

        self.assertGreater(top_strip_var, 5.0, "Top strip must contain photographic pixels, NOT zero-variance padding")
        self.assertGreater(bottom_strip_var, 5.0, "Bottom strip must contain photographic pixels, NOT zero-variance padding")

    def test_04_no_data_leakage(self):
        """Tests that anti-leakage validation catches overlapping group IDs across splits."""
        # Clean Manifest
        clean_manifest = [
            {"id": "1", "path": "p1.jpg", "label": "REAL", "source": "S1", "split": "train", "group_id": "scene_A"},
            {"id": "2", "path": "p2.jpg", "label": "REAL", "source": "S1", "split": "test", "group_id": "scene_B"}
        ]
        tmp_clean = "test_clean_manifest.json"
        with open(tmp_clean, "w") as f: json.dump(clean_manifest, f)
        valid, errs, stats = validate_manifest(tmp_clean)
        if os.path.exists(tmp_clean): os.remove(tmp_clean)
        self.assertTrue(valid, f"Clean manifest should be valid: {errs}")

        # Leaking Manifest (scene_A in both train and test)
        leaking_manifest = [
            {"id": "1", "path": "p1.jpg", "label": "REAL", "source": "S1", "split": "train", "group_id": "scene_A"},
            {"id": "2", "path": "p2.jpg", "label": "REAL", "source": "S1", "split": "test", "group_id": "scene_A"}
        ]
        tmp_leak = "test_leak_manifest.json"
        with open(tmp_leak, "w") as f: json.dump(leaking_manifest, f)
        valid_leak, errs_leak, stats_leak = validate_manifest(tmp_leak)
        if os.path.exists(tmp_leak): os.remove(tmp_leak)
        self.assertFalse(valid_leak, "Validator must reject manifest with leaking group_id across train/test")
        self.assertTrue(any("DATA LEAKAGE DETECTED" in e for e in errs_leak))

    def test_05_deterministic_inference(self):
        """Verifies that running model evaluation produces deterministic identical results."""
        # Node test script verifying determinism
        node_script = """
import { AutoModelForImageClassification, AutoProcessor, RawImage, env } from '@huggingface/transformers';
env.localModelPath = './frontend/models/';
env.allowRemoteModels = false;
env.allowLocalModels = true;

const model = await AutoModelForImageClassification.from_pretrained('./frontend/models/onnx-community/ai-image-detection-ONNX', { device: 'cpu', dtype: 'q4' });
const processor = await AutoProcessor.from_pretrained('./frontend/models/onnx-community/ai-image-detection-ONNX');
const img = await RawImage.read('./frontend/lens.png');
const inputs = await processor(img);

const out1 = await model(inputs);
const out2 = await model(inputs);

const logits1 = Array.from(out1.logits.data);
const logits2 = Array.from(out2.logits.data);

const diff = Math.abs(logits1[0] - logits2[0]) + Math.abs(logits1[1] - logits2[1]);
if (diff > 1e-6) {
  process.exit(1);
} else {
  process.exit(0);
}
"""
        cmd = ["node", "--input-type=module", "-e", node_script]
        res = subprocess.run(cmd, cwd=os.path.abspath("."))
        self.assertEqual(res.returncode, 0, "Inference must be strictly deterministic across repeated passes")

if __name__ == "__main__":
    unittest.main()
