"""
TRUSTLENS Phase 20: Real-World Robustness Suite
Tests perturbation stability across:
1. Native original
2. Resized (50%)
3. JPEG compressed (Q50)
4. Gaussian blurred (radius 1.0)
5. Brightness scaled (+20%)
6. Contrast scaled (+20%)
"""

import sys
from pathlib import Path
import torch
from PIL import Image, ImageEnhance, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.evaluate_finetuned import load_model_from_checkpoint, get_eval_transform

def run_robustness_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("TRUSTLENS REAL-WORLD ROBUSTNESS & PERTURBATION TEST")
    print(f"Device: {device}")
    print("=" * 75)

    fine_ckpt = PROJECT_ROOT / "training" / "checkpoints" / "best_val.safetensors"
    model = load_model_from_checkpoint(fine_ckpt, device)
    transform = get_eval_transform()

    def get_score(img):
        t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(t).item()
            prob = torch.sigmoid(torch.tensor(logit)).item()
        return round(prob * 100.0, 1)

    # Test cases: 1 real photo, 1 AI image
    test_cases = [
        ("Real Photo (Messi)", PROJECT_ROOT / "training" / "data" / "raw" / "real_messi.jpg", "REAL"),
        ("AI Image (DALL-E 2)", PROJECT_ROOT / "training" / "data" / "raw" / "ai_dalle2_01.png", "AI")
    ]

    for name, img_path, truth in test_cases:
        base_img = Image.open(img_path).convert("RGB")
        
        # Perturbations
        variants = {
            "Original Unmodified": base_img,
            "Resized (-50% bilinear)": base_img.resize((base_img.width // 2, base_img.height // 2)),
            "JPEG Compressed (Q50)": None, # Will compress to buffer
            "Gaussian Blurred (r=1.0)": base_img.filter(ImageFilter.GaussianBlur(radius=1.0)),
            "Brightness Altered (+20%)": ImageEnhance.Brightness(base_img).enhance(1.2),
            "Contrast Altered (+20%)": ImageEnhance.Contrast(base_img).enhance(1.2)
        }

        # JPEG Q50
        import io
        buf = io.BytesIO()
        base_img.save(buf, format="JPEG", quality=50)
        variants["JPEG Compressed (Q50)"] = Image.open(buf)

        print(f"\n--- {name} (Truth: {truth}) ---")
        base_score = get_score(base_img)
        print(f"  {'Perturbation':<28} | {'Synthetic Score':<16} | {'Classification'}")
        print("  " + "-" * 55)

        for p_name, p_img in variants.items():
            s = get_score(p_img)
            label = "AI" if s >= 50.0 else "REAL"
            status = "STABLE" if (label == truth) else "SHIFTED"
            print(f"  {p_name:<28} | {s:>14.1f}% | {label:<5} ({status})")

    print("\n" + "=" * 75)
    print("ROBUSTNESS TEST COMPLETE")
    print("=" * 75)

if __name__ == "__main__":
    run_robustness_test()
