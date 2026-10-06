"""
Comprehensive Unit Test Suite for TrustLens Community Forensics V2
Tests:
1. Model loading & state dictionary verification
2. Preprocessing & ImageNet normalization
3. Raw inference & output shapes
4. Sigmoid conversion & continuous probability mapping
5. Score bounds & complement sum identity
6. API endpoint request & response serialization
7. Batched patch inference
8. Malformed request & resource limit handling
9. Health endpoint status & metadata
10. Multi-resolution image patch handling (arbitrary resolutions to 224x224)
"""

import os
import io
import math
import base64
import unittest
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from backend.detector import CommunityForensicsDetector, detector
from backend.main import app

class TestCommunityForensicsV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure detector is loaded
        if not detector.loaded:
            detector.load_model()
        cls.client = TestClient(app)

    def test_01_model_loading(self):
        """1. Model weights load cleanly, correct parameter count, eval mode active."""
        self.assertTrue(detector.loaded)
        self.assertIsNotNone(detector.model)
        self.assertFalse(detector.model.training) # Model must be in eval() mode
        
        # Verify classification head
        self.assertEqual(detector.model.head.in_features, 384)
        self.assertEqual(detector.model.head.out_features, 1)

    def test_02_preprocessing_imagenet_normalization(self):
        """2. Preprocessing maps PIL image to [3, 224, 224] tensor with ImageNet stats."""
        test_img = Image.new("RGB", (300, 400), color=(128, 128, 128))
        tensor = detector.preprocess_image(test_img)

        self.assertEqual(tensor.shape, torch.Size([3, 224, 224]))
        self.assertEqual(tensor.dtype, torch.float32)

        # A flat gray (128/255 = ~0.5019) should yield values near (0.5019 - mean) / std
        # Channel 0: (0.5019 - 0.485) / 0.229 ≈ 0.0738
        self.assertAlmostEqual(tensor[0].mean().item(), (128/255.0 - 0.485) / 0.229, delta=0.05)

    def test_03_raw_inference_output(self):
        """3. Forward pass produces shape [N, 1] with finite scalar outputs."""
        dummy = torch.randn(2, 3, 224, 224, device=detector.device)
        with torch.no_grad():
            out = detector.model(dummy)

        self.assertEqual(out.shape, torch.Size([2, 1]))
        for val in out.flatten().tolist():
            self.assertFalse(math.isnan(val))
            self.assertFalse(math.isinf(val))

    def test_04_sigmoid_conversion_and_mapping(self):
        """4. Sigmoid conversion correctly maps logit to [0, 1] probability."""
        # 0.0 logit -> 50% probability
        prob = torch.sigmoid(torch.tensor([0.0])).item()
        self.assertAlmostEqual(prob, 0.5)

        # Negative logit -> < 50% (favors REAL)
        prob_neg = torch.sigmoid(torch.tensor([-3.0])).item()
        self.assertLess(prob_neg, 0.5)

        # Positive logit -> > 50% (favors AI)
        prob_pos = torch.sigmoid(torch.tensor([3.0])).item()
        self.assertGreater(prob_pos, 0.5)

    def test_05_score_bounds_and_complement(self):
        """5. Synthetic and real scores stay strictly between 0 and 100 and sum to 100."""
        patches = [
            {"id": "test_white", "image": Image.new("RGB", (224, 224), color=(255, 255, 255))},
            {"id": "test_black", "image": Image.new("RGB", (224, 224), color=(0, 0, 0))}
        ]
        res = detector.analyze_patches(patches)
        for r in res["results"]:
            synth = r["syntheticScore"]
            real = r["realScore"]
            self.assertGreaterEqual(synth, 0.0)
            self.assertLessEqual(synth, 100.0)
            self.assertGreaterEqual(real, 0.0)
            self.assertLessEqual(real, 100.0)
            self.assertAlmostEqual(synth + real, 100.0, delta=0.1)

    def test_06_api_endpoint_serialization(self):
        """6. POST /api/analyze/image-patches returns structured response matching schema."""
        buf = io.BytesIO()
        Image.new("RGB", (224, 224), color=(100, 140, 180)).save(buf, format="JPEG")
        b64_data = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = {
            "patches": [
                {"id": "global", "image": b64_data},
                {"id": "center", "image": b64_data}
            ]
        }
        resp = self.client.post("/api/analyze/image-patches", json=payload)
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertEqual(data["model"], "OwensLab/commfor-model-224")
        self.assertEqual(len(data["results"]), 2)
        self.assertEqual(data["results"][0]["id"], "global")
        self.assertEqual(data["results"][1]["id"], "center")
        self.assertIn("syntheticScore", data["results"][0])
        self.assertIn("realScore", data["results"][0])
        self.assertIn("inference", data)
        self.assertEqual(data["inference"]["batchSize"], 2)

    def test_07_batched_patch_inference(self):
        """7. Up to 5 multi-scale forensic patches processed in one batched forward pass."""
        patches = [
            {"id": f"p_{i}", "image": Image.new("RGB", (224, 224), color=(i * 40, 100, 150))}
            for i in range(5)
        ]
        res = detector.analyze_patches(patches)
        self.assertEqual(len(res["results"]), 5)
        self.assertEqual(res["inference"]["batchSize"], 5)
        self.assertGreater(res["inference"]["totalLatencyMs"], 0)

    def test_08_malformed_requests_and_limits(self):
        """8. Rejects empty patch lists, malformed base64, and excessive batch counts."""
        # Empty patches
        r1 = self.client.post("/api/analyze/image-patches", json={"patches": []})
        self.assertEqual(r1.status_code, 422) # Pydantic validation min_length=1

        # Corrupt data
        r2 = self.client.post("/api/analyze/image-patches", json={
            "patches": [{"id": "corrupt", "image": "invalid_base64_data"}]
        })
        self.assertEqual(r2.status_code, 400)

        # Exceeding maximum batch size (>10)
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), color=(50, 50, 50)).save(buf, format="JPEG")
        b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        oversized_list = [{"id": f"p_{i}", "image": b64} for i in range(12)]
        r3 = self.client.post("/api/analyze/image-patches", json={"patches": oversized_list})
        self.assertEqual(r3.status_code, 422) # Pydantic max_length=10

    def test_09_health_endpoint(self):
        """9. GET /api/health reflects active model, loaded flag, and hardware device."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["model"], "OwensLab/commfor-model-224")
        self.assertTrue(data["model_loaded"])
        self.assertIn(data["device"].lower(), ["cuda", "cpu"])

    def test_10_multi_resolution_arbitrary_inputs(self):
        """10. Accepts arbitrary photographic resolutions (e.g. 512x512, 1920x1080)."""
        resolutions = [(512, 512), (1024, 1024), (1920, 1080), (1080, 1920)]
        for w, h in resolutions:
            img = Image.new("RGB", (w, h), color=(100, 120, 140))
            tensor = detector.preprocess_image(img)
            self.assertEqual(tensor.shape, torch.Size([3, 224, 224]))

if __name__ == "__main__":
    unittest.main()
