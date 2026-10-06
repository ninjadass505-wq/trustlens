"""
TrustLens Forensic Detector Engine
Pretrained Model: OwensLab/commfor-model-224 (Community Forensics)
Architecture: timm ViT-Small (vit_small_patch16_224.augreg_in21k_ft_in1k) + Linear(384 -> 1)
"""

import os
import io
import time
import math
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import timm
from safetensors.torch import load_file

logger = logging.getLogger("trustlens.detector")

# Standard ImageNet normalization parameters for ViT
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class CommunityForensicsDetector:
    def __init__(self, model_dir: str | Path | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "OwensLab/commfor-model-224"
        self.model_version = "commfor-model-224"
        self.loaded = False
        self.model = None

        active_version = os.environ.get("TRUSTLENS_MODEL_VERSION", "finetuned").lower()
        if model_dir is None:
            models_root = Path(__file__).resolve().parent / "models"
            if active_version == "original" and (models_root / "original" / "commfor_model_original.safetensors").exists():
                self.model_path = models_root / "original" / "commfor_model_original.safetensors"
                self.model_version = "commfor-model-224-original"
            elif (models_root / "finetuned" / "model.safetensors").exists():
                self.model_path = models_root / "finetuned" / "model.safetensors"
                self.model_version = "commfor-model-224-finetuned"
            else:
                self.model_path = models_root / "commfor-model-224" / "model.safetensors"
                self.model_version = "commfor-model-224"
        else:
            self.model_path = Path(model_dir) / "model.safetensors"

        # Preprocessing pipeline: convert PIL RGB to [3, 224, 224] with ImageNet normalization
        self.transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def load_model(self) -> None:
        """Loads and verifies the model weights once at startup."""
        print("[TRUSTLENS] Loading Community Forensics 224...")
        print(f"[TRUSTLENS] Device: {str(self.device).upper()}")

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"[TRUSTLENS] Model weights not found at {self.model_path}. "
                "Ensure model.safetensors is located in backend/models/commfor-model-224/"
            )

        # 1. Instantiate architecture: vit_small_patch16_224 with 1-output linear head
        model = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=False)
        model.head = nn.Linear(in_features=384, out_features=1, bias=True)

        # 2. Load weights from safetensors
        raw_state = load_file(str(self.model_path))
        # Strip 'vit.' prefix if present from PyTorchModelHubMixin format
        clean_state = {
            (k[4:] if k.startswith("vit.") else k): v
            for k, v in raw_state.items()
        }
        missing, unexpected = model.load_state_dict(clean_state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"State dict mismatch! Missing: {missing}, Unexpected: {unexpected}")

        model.to(self.device)
        model.eval()
        self.model = model

        # 3. Startup sanity validation (Requirement 21)
        self._validate_startup_sanity()

        self.loaded = True
        print("[TRUSTLENS] Model loaded successfully")
        print("[TRUSTLENS] Input: 224x224 RGB")
        print("[TRUSTLENS] Output: single synthetic logit")

    def _validate_startup_sanity(self) -> None:
        """Runs a tiny sanity check to ensure output shapes, bounds, and mathematics are valid."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            out = self.model(dummy)

            if out.shape != torch.Size([1, 1]):
                raise ValueError(f"Sanity check failed: Expected shape [1, 1], got {out.shape}")

            val = out.item()
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Sanity check failed: Model returned non-finite output {val}")

            prob = torch.sigmoid(out).item()
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"Sanity check failed: Sigmoid probability out of bounds: {prob}")

            synthetic_score = prob * 100.0
            real_score = 100.0 - synthetic_score
            if abs((synthetic_score + real_score) - 100.0) > 1e-4:
                raise ValueError("Sanity check failed: Synthetic and real scores do not sum to 100")

    def preprocess_image(self, img: Image.Image) -> torch.Tensor:
        """Converts a PIL image to a normalized [3, 224, 224] PyTorch tensor."""
        rgb_img = img.convert("RGB")
        return self.transform(rgb_img)

    def analyze_patches(self, patches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes batched inference across all submitted forensic patches in ONE model call.
        patches format: [ {"id": "global", "image": PIL.Image}, ... ]
        """
        if not self.loaded or self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        if not patches:
            raise ValueError("No patches provided for forensic analysis.")

        if len(patches) > 10:
            raise ValueError("Exceeded maximum batch limit of 10 patches.")

        t_start = time.perf_counter()

        # Step 1: Preprocess all patches into a batched tensor [N, 3, 224, 224]
        t0_pre = time.perf_counter()
        tensors = []
        for p in patches:
            img = p["image"]
            tensor = self.preprocess_image(img)
            tensors.append(tensor)

        batch_tensor = torch.stack(tensors, dim=0).to(self.device)
        t_pre = (time.perf_counter() - t0_pre) * 1000.0

        # Step 2: Batched model forward pass
        t0_inf = time.perf_counter()
        with torch.no_grad():
            logits = self.model(batch_tensor) # [N, 1]
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy() # [N]
        t_inf = (time.perf_counter() - t0_inf) * 1000.0

        total_latency = (time.perf_counter() - t_start) * 1000.0

        # Step 3: Format per-patch scores
        results = []
        for i, p in enumerate(patches):
            p_id = p.get("id", f"patch_{i}")
            synthetic_prob = float(probs[i])
            synthetic_score = round(synthetic_prob * 100.0, 1)
            real_score = round(100.0 - synthetic_score, 1)

            results.append({
                "id": p_id,
                "syntheticScore": synthetic_score,
                "realScore": real_score,
                "rawProbability": round(synthetic_prob, 4)
            })

        return {
            "model": self.model_name,
            "model_version": self.model_version,
            "results": results,
            "inference": {
                "device": str(self.device),
                "batchSize": len(patches),
                "preprocessingMs": round(t_pre, 2),
                "inferenceMs": round(t_inf, 2),
                "totalLatencyMs": round(total_latency, 2)
            }
        }

# Global singleton detector instance
detector = CommunityForensicsDetector()
