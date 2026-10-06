"""
TRUSTLENS Instant Rollback & Model Switcher Utility
Allows instant switching between:
- 'finetuned': The improved fine-tuned checkpoint (backend/models/finetuned/model.safetensors)
- 'original': The original baseline checkpoint (backend/models/original/commfor_model_original.safetensors)

Usage:
  python backend/rollback_model.py            # Rolls back to original baseline
  python backend/switch_model.py --version finetuned # Activates fine-tuned model
"""

import sys
import shutil
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"

def rollback_to_original():
    print("=" * 65)
    print("TRUSTLENS: ROLLING BACK TO ORIGINAL BASELINE MODEL")
    print("=" * 65)

    orig_path = MODELS_DIR / "original" / "commfor_model_original.safetensors"
    active_path = MODELS_DIR / "commfor-model-224" / "model.safetensors"

    if not orig_path.exists():
        print(f"Error: Original checkpoint not found at {orig_path}")
        sys.exit(1)

    # Copy original to active
    shutil.copy2(orig_path, active_path)
    print(f"Restored original baseline checkpoint to: {active_path}")
    print("Rollback COMPLETE. The original detector is now active.")
    print("=" * 65)

def activate_finetuned():
    print("=" * 65)
    print("TRUSTLENS: ACTIVATING FINE-TUNED MODEL CHECKPOINT")
    print("=" * 65)

    fine_path = MODELS_DIR / "finetuned" / "model.safetensors"
    active_path = MODELS_DIR / "commfor-model-224" / "model.safetensors"

    if not fine_path.exists():
        print(f"Error: Fine-tuned checkpoint not found at {fine_path}")
        sys.exit(1)

    shutil.copy2(fine_path, active_path)
    print(f"Activated fine-tuned checkpoint at: {active_path}")
    print("Activation COMPLETE. The fine-tuned detector is now active.")
    print("=" * 65)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() in ["--finetuned", "-f", "finetuned"]:
        activate_finetuned()
    else:
        rollback_to_original()
