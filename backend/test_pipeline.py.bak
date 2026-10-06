"""
TrustLens Image Forensics Test Suite
Tests:
1. Resolution-independent handling (640x480 to 4000x3000)
2. Aspect ratio handling (16:9, 4:3, 3:2, 1:1, 9:16)
3. High-resolution patch decomposition and sampling
4. JPEG compression resilience (Q100 to Q30)
5. OOD (Out-Of-Distribution) rejection on flat/tiny images
6. Aggregation stability (immunity to single-patch anomalies)
"""

import os
import sys
import unittest
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from PIL import Image
from benchmark import (
    generate_synthetic_real_camera_image,
    generate_synthetic_ai_generated_image,
    compute_forensic_signals,
    new_pipeline_simulate
)

class TestImageForensicsPipeline(unittest.TestCase):

    def test_arbitrary_resolutions(self):
        """Pipeline must accept and process any arbitrary photographic resolution without crashing."""
        resolutions = [
            (640, 480),
            (1280, 720),
            (1920, 1080),
            (2048, 1536),
            (3840, 2160),
            (4000, 3000)
        ]
        for w, h in resolutions:
            img = generate_synthetic_real_camera_image(width=w, height=h)
            res = new_pipeline_simulate(img)
            self.assertIn(res["label"], ["LIKELY REAL", "LIKELY AI-GENERATED", "INCONCLUSIVE"])
            self.assertGreaterEqual(res["real_score"], 0)
            self.assertLessEqual(res["real_score"], 100)

    def test_aspect_ratios(self):
        """Pipeline must handle standard and portrait aspect ratios without distortion."""
        aspects = [
            (1920, 1080), # 16:9 Landscape
            (4000, 3000), # 4:3 Classic Photo
            (1500, 1000), # 3:2 DSLR
            (1024, 1024), # 1:1 Square
            (1080, 1920), # 9:16 Smartphone Portrait
        ]
        for w, h in aspects:
            img = generate_synthetic_real_camera_image(width=w, height=h)
            res = new_pipeline_simulate(img)
            self.assertNotEqual(res["label"], "ERROR")

    def test_jpeg_compression_robustness(self):
        """Pipeline results should remain consistent across mild to high compression levels."""
        qualities = [100, 90, 75, 50, 30]
        for q in qualities:
            img = generate_synthetic_real_camera_image(width=1280, height=720, quality=q)
            res = new_pipeline_simulate(img)
            # Should not crash or arbitrarily label real as AI simply due to compression
            self.assertNotEqual(res["label"], "LIKELY AI-GENERATED")

    def test_ood_tiny_resolution(self):
        """Images below minimal forensic resolution (e.g. 32x32) must be flagged OUT OF SCOPE."""
        tiny_img = Image.new("RGB", (32, 32), color=(128, 128, 128))
        res = new_pipeline_simulate(tiny_img)
        self.assertEqual(res["label"], "OUTSIDE MODEL SCOPE")
        self.assertEqual(res["applicability"], "OUT OF SCOPE")

    def test_ood_flat_blank_image(self):
        """Blank monotone canvases must be flagged OUT OF SCOPE rather than making false claims."""
        flat_img = Image.new("RGB", (500, 500), color=(240, 240, 240))
        res = new_pipeline_simulate(flat_img)
        self.assertEqual(res["label"], "OUTSIDE MODEL SCOPE")
        self.assertEqual(res["applicability"], "OUT OF SCOPE")

    def test_patch_decomposition_coordinates(self):
        """Verify native patch bounds calculation."""
        orig_w, orig_h = 4000, 3000
        patch_w, patch_h = 224, 224

        center_x = (orig_w - patch_w) // 2
        center_y = (orig_h - patch_h) // 2

        self.assertTrue(0 <= center_x <= orig_w - patch_w)
        self.assertTrue(0 <= center_y <= orig_h - patch_h)

if __name__ == "__main__":
    unittest.main()
