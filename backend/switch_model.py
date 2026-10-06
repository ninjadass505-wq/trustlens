"""
TRUSTLENS Model Switcher Utility
Usage:
  python backend/switch_model.py --version finetuned
  python backend/switch_model.py --version original
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from backend.rollback_model import rollback_to_original, activate_finetuned
except ImportError:
    from rollback_model import rollback_to_original, activate_finetuned

def main():
    parser = argparse.ArgumentParser(description="Switch active TRUSTLENS model checkpoint")
    parser.add_argument("--version", choices=["finetuned", "original"], default="finetuned",
                        help="Checkpoint version to activate ('finetuned' or 'original')")
    args = parser.parse_args()

    if args.version == "original":
        rollback_to_original()
    else:
        activate_finetuned()

if __name__ == "__main__":
    main()
