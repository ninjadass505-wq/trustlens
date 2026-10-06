"""
TRUSTLENS Phase 13: Inference Regression Test
Verifies numerical consistency between:
1. PyTorch fine-tuned model checkpoint
2. Production detector runtime (backend.detector)
3. Live FastAPI endpoint serialization
"""

import io
import json
import base64
import unittest
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from training.evaluate_finetuned import load_model_from_checkpoint, get_eval_transform
from backend.detector import detector
from backend.main import app

class TestInferenceRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.fine_ckpt = PROJECT_ROOT / "backend" / "models" / "finetuned" / "model.safetensors"
        cls.raw_model = load_model_from_checkpoint(cls.fine_ckpt, cls.device)
        detector.load_model()
        cls.client = TestClient(app)
        cls.transform = get_eval_transform()

    def test_numerical_equivalence(self):
        """Direct PyTorch evaluation must exactly match backend detector and API endpoint."""
        test_img = Image.new("RGB", (224, 224), color=(140, 160, 180))

        # 1. Raw PyTorch
        t = self.transform(test_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.raw_model(t).item()
            raw_synth = torch.sigmoid(torch.tensor(logit)).item() * 100.0

        # 2. Production Detector
        det_res = detector.analyze_patches([{"id": "test_patch", "image": test_img}])
        det_synth = det_res["results"][0]["syntheticScore"]

        # 3. FastAPI Client
        buf = io.BytesIO()
        test_img.save(buf, format="JPEG")
        b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        api_res = self.client.post("/api/analyze/image-patches", json={"patches": [{"id": "test_patch", "image": b64}]})
        api_synth = api_res.json()["results"][0]["syntheticScore"]

        print(f"Raw PyTorch:        {raw_synth:.2f}%")
        print(f"Detector Engine:    {det_synth:.2f}%")
        print(f"FastAPI Endpoint:   {api_synth:.2f}%")

        # Must be practically identical (<0.5% tolerance due to JPEG round-trip)
        self.assertAlmostEqual(raw_synth, det_synth, delta=0.1)
        self.assertAlmostEqual(det_synth, api_synth, delta=0.5)
        print("Inference Regression Test PASSED: Numerical Equivalence Verified!")

if __name__ == "__main__":
    unittest.main()
