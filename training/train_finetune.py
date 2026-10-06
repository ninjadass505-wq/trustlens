"""
TRUSTLENS Phase 6, 7, 8 & 25: Conservative Two-Stage Fine-Tuner

Fine-tunes the EXISTING Community Forensics checkpoint:
- Starting Checkpoint: backend/models/original/commfor_model_original.safetensors
- Architecture: timm vit_small_patch16_224.augreg_in21k_ft_in1k + Linear(384 -> 1)
- Loss: BCEWithLogitsLoss() (proper un-sigmoid logit loss)
- Two-Stage Approach:
    Stage 1: Freeze ViT backbone, fine-tune Linear head only (LR: 1e-4)
    Stage 2: Differential unfreezing of final layers (Backbone LR: 5e-6, Head LR: 2e-5)
- Validation tracking, early stopping, and checkpointing
- Saves history to training/training_history.json
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import timm
from safetensors.torch import load_file, save_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.detector import IMAGENET_MEAN, IMAGENET_STD
from training.evaluate_finetuned import evaluate_dataset, compute_roc_auc

# Set deterministic random seed
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

class ForensicDataset(Dataset):
    def __init__(self, split_dir: Path, transform=None):
        self.samples: List[Tuple[Path, float]] = []
        self.transform = transform

        # Label 0: Real
        real_files = sorted(list((split_dir / "real").glob("*")))
        for p in real_files:
            self.samples.append((p, 0.0))

        # Label 1: AI
        ai_files = sorted(list((split_dir / "ai").glob("*")))
        for p in ai_files:
            self.samples.append((p, 1.0))

        # Deterministic shuffle
        random.Random(SEED).shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor([label], dtype=torch.float32)

def get_train_transforms():
    """Conservative forensic augmentations: preserves forensic signals without distortion."""
    return T.Compose([
        T.RandomResizedCrop(224, scale=(0.85, 1.0), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.08, contrast=0.08),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("TRUSTLENS CONSERVATIVE TWO-STAGE FINE-TUNING")
    print(f"Device: {device}")
    print("=" * 75)

    checkpoints_dir = PROJECT_ROOT / "training" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load starting production model
    orig_ckpt = PROJECT_ROOT / "backend" / "models" / "original" / "commfor_model_original.safetensors"
    print(f"Loading Base Checkpoint: {orig_ckpt}")
    
    model = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=False)
    model.head = nn.Linear(in_features=384, out_features=1, bias=True)
    raw_state = load_file(str(orig_ckpt))
    clean_state = {(k[4:] if k.startswith("vit.") else k): v for k, v in raw_state.items()}
    model.load_state_dict(clean_state, strict=True)
    model.to(device)

    # Datasets
    data_dir = PROJECT_ROOT / "training" / "data"
    train_dataset = ForensicDataset(data_dir / "train", transform=get_train_transforms())
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    val_dir = data_dir / "val"
    criterion = nn.BCEWithLogitsLoss()

    history = {
        "stage1": [],
        "stage2": [],
        "best_epoch": None,
        "best_val_f1": 0.0,
        "best_val_accuracy": 0.0,
        "best_val_loss": 999.0
    }

    best_checkpoint_path = checkpoints_dir / "best_val.safetensors"
    best_val_metric = -1.0 # Combined F1 + RealAcc objective

    # -------------------------------------------------------------
    # STAGE 1: FREEZE BACKBONE, TRAIN HEAD ONLY
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("STAGE 1: Linear Head Fine-Tuning (Backbone Frozen)")
    print("-" * 75)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True

    optimizer_s1 = torch.optim.AdamW(model.head.parameters(), lr=2e-4, weight_decay=1e-2)

    for epoch in range(1, 7):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer_s1.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer_s1.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_res = evaluate_dataset(model, val_dir, device)
        
        # Target metric: High Real Accuracy (>=90%) while maximizing F1
        composite_score = val_res["f1"] + val_res["real_accuracy"]
        print(f"Stage 1 [Epoch {epoch:02d}/06] Train Loss: {avg_train_loss:.4f} | Val Acc: {val_res['accuracy']:.1f}% | Real Acc: {val_res['real_accuracy']:.1f}% | AI Acc: {val_res['ai_accuracy']:.1f}% | F1: {val_res['f1']:.1f}%")

        epoch_record = {
            "stage": 1,
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_metrics": val_res
        }
        history["stage1"].append(epoch_record)

        if composite_score > best_val_metric:
            best_val_metric = composite_score
            history["best_epoch"] = f"stage1_epoch_{epoch}"
            history["best_val_f1"] = val_res["f1"]
            history["best_val_accuracy"] = val_res["accuracy"]
            # Save checkpoint
            state_to_save = {f"vit.{k}": v for k, v in model.state_dict().items()}
            save_file(state_to_save, str(best_checkpoint_path))

    # -------------------------------------------------------------
    # STAGE 2: SUBTLE DEEP LAYER UNFREEZING
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("STAGE 2: Differential Fine-Tuning (Unfreezing Blocks 10, 11 + Head)")
    print("-" * 75)

    # Unfreeze only the last 2 transformer blocks and norm
    for name, param in model.named_parameters():
        if "blocks.10" in name or "blocks.11" in name or "norm" in name or "head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and "head" not in n]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "head" in n]

    optimizer_s2 = torch.optim.AdamW([
        {"params": backbone_params, "lr": 5e-6},
        {"params": head_params, "lr": 2e-5}
    ], weight_decay=1e-2)

    for epoch in range(1, 9):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer_s2.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_s2.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_res = evaluate_dataset(model, val_dir, device)
        composite_score = val_res["f1"] + val_res["real_accuracy"]
        print(f"Stage 2 [Epoch {epoch:02d}/08] Train Loss: {avg_train_loss:.4f} | Val Acc: {val_res['accuracy']:.1f}% | Real Acc: {val_res['real_accuracy']:.1f}% | AI Acc: {val_res['ai_accuracy']:.1f}% | F1: {val_res['f1']:.1f}%")

        epoch_record = {
            "stage": 2,
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_metrics": val_res
        }
        history["stage2"].append(epoch_record)

        # Ensure Real Accuracy never collapses below 100%
        if val_res["real_accuracy"] == 100.0 and val_res["f1"] >= history["best_val_f1"]:
            best_val_metric = composite_score
            history["best_epoch"] = f"stage2_epoch_{epoch}"
            history["best_val_f1"] = val_res["f1"]
            history["best_val_accuracy"] = val_res["accuracy"]
            state_to_save = {f"vit.{k}": v for k, v in model.state_dict().items()}
            save_file(state_to_save, str(best_checkpoint_path))
            print(f"  --> [NEW BEST CHECKPOINT SAVED] Val Acc: {val_res['accuracy']}%, F1: {val_res['f1']}%")

    # Save training history
    history_path = PROJECT_ROOT / "training" / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 75)
    print(f"TRAINING COMPLETE. Best Epoch: {history['best_epoch']} (Val Acc: {history['best_val_accuracy']}%, Val F1: {history['best_val_f1']}%)")
    print(f"Best checkpoint saved at: {best_checkpoint_path}")
    print(f"Training history saved at: {history_path}")
    print("=" * 75)

if __name__ == "__main__":
    train()
