"""
TRUSTLENS Benchmark Manifest Validator & Anti-Leakage Verifier
Ensures dataset integrity, schema compliance, and strictly zero data leakage across splits.
"""

import json
import os
import sys
from typing import Dict, List, Tuple, Set

REQUIRED_FIELDS = [
    "id",
    "path",
    "label",
    "source",
    "split"
]

OPTIONAL_FIELDS = [
    "generator",
    "generator_family",
    "camera_device",
    "resolution",
    "aspect_ratio",
    "compression",
    "scene_category",
    "group_id"
]

VALID_LABELS = {"REAL", "SYNTHETIC"}
VALID_SPLITS = {"train", "validation", "test"}

class ManifestValidationError(Exception):
    pass

def validate_manifest(manifest_path: str, dataset_root: str = "") -> Tuple[bool, List[str], Dict]:
    """
    Validates a dataset manifest against schema and strict leakage prevention rules.
    Returns (is_valid, errors, summary_statistics).
    """
    errors = []
    
    if not os.path.exists(manifest_path):
        return False, [f"Manifest file not found: {manifest_path}"], {}

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, [f"JSON parse error in manifest: {str(e)}"], {}

    if not isinstance(data, list):
        return False, ["Manifest root must be a JSON array of image objects."], {}

    seen_ids: Set[str] = set()
    seen_paths: Set[str] = set()

    split_groups: Dict[str, Set[str]] = {s: set() for s in VALID_SPLITS}
    split_counts: Dict[str, Dict[str, int]] = {s: {"REAL": 0, "SYNTHETIC": 0} for s in VALID_SPLITS}
    generator_splits: Dict[str, Set[str]] = {}
    resolution_counts: Dict[str, int] = {}
    generators: Set[str] = set()

    for i, item in enumerate(data):
        item_id = item.get("id")
        # 1. Required fields check
        for rf in REQUIRED_FIELDS:
            if rf not in item or item[rf] is None or item[rf] == "":
                errors.append(f"Item #{i} (ID: {item_id}) missing required field '{rf}'")

        if item_id:
            if item_id in seen_ids:
                errors.append(f"Duplicate image ID: '{item_id}'")
            seen_ids.add(item_id)

        # 2. Path verification
        img_path = item.get("path", "")
        if img_path:
            if img_path in seen_paths:
                errors.append(f"Duplicate image path: '{img_path}'")
            seen_paths.add(img_path)

            if dataset_root:
                full_path = os.path.join(dataset_root, img_path)
                if not os.path.exists(full_path):
                    errors.append(f"File path does not exist on disk: '{full_path}'")

        # 3. Label verification
        label = item.get("label", "").upper()
        if label not in VALID_LABELS:
            errors.append(f"Invalid label '{label}' for ID '{item_id}'. Must be REAL or SYNTHETIC.")

        # 4. Split verification
        split = item.get("split", "").lower()
        if split not in VALID_SPLITS:
            errors.append(f"Invalid split '{split}' for ID '{item_id}'. Must be train, validation, or test.")
        else:
            if label in VALID_LABELS:
                split_counts[split][label] += 1

        # 5. Generator consistency
        gen = item.get("generator")
        if label == "SYNTHETIC":
            if not gen:
                errors.append(f"Synthetic item '{item_id}' must specify a 'generator' name.")
            else:
                generators.add(gen)
                if gen not in generator_splits:
                    generator_splits[gen] = set()
                generator_splits[gen].add(split)
        elif label == "REAL" and gen:
            errors.append(f"Real photograph '{item_id}' must have generator=null, but got '{gen}'.")

        # 6. Group ID / Leakage tracking
        group_id = item.get("group_id")
        if group_id and split in VALID_SPLITS:
            split_groups[split].add(group_id)

        # 7. Metadata stats
        res = item.get("resolution", "unknown")
        resolution_counts[res] = resolution_counts.get(res, 0) + 1

    # ========================================================
    # ANTI-LEAKAGE VERIFICATION:
    # 1. No group_id (scene/burst/prompt cluster) may cross splits
    # ========================================================
    train_test_overlap = split_groups["train"].intersection(split_groups["test"])
    if train_test_overlap:
        errors.append(f"DATA LEAKAGE DETECTED: {len(train_test_overlap)} group_id(s) appear in BOTH train and test splits! Example: {list(train_test_overlap)[:3]}")

    train_val_overlap = split_groups["train"].intersection(split_groups["validation"])
    if train_val_overlap:
        errors.append(f"DATA LEAKAGE DETECTED: {len(train_val_overlap)} group_id(s) appear in BOTH train and validation splits! Example: {list(train_val_overlap)[:3]}")

    val_test_overlap = split_groups["validation"].intersection(split_groups["test"])
    if val_test_overlap:
        errors.append(f"DATA LEAKAGE DETECTED: {len(val_test_overlap)} group_id(s) appear in BOTH validation and test splits! Example: {list(val_test_overlap)[:3]}")

    summary = {
        "total_images": len(data),
        "unique_ids": len(seen_ids),
        "split_counts": split_counts,
        "generators": sorted(list(generators)),
        "generator_split_distribution": {k: sorted(list(v)) for k, v in generator_splits.items()},
        "leakage_clean": len(train_test_overlap) == 0 and len(train_val_overlap) == 0 and len(val_test_overlap) == 0
    }

    return (len(errors) == 0), errors, summary

def create_sample_manifest(output_path: str):
    """
    Creates a benchmark template manifest with representative real-world & synthetic entries.
    """
    sample = [
        # --- TEST SPLIT: REAL CAMERA CAPTURES ---
        {
            "id": "real_cam_001",
            "path": "test/real/real_cam_001.jpg",
            "label": "REAL",
            "source": "RAISE",
            "generator": None,
            "generator_family": None,
            "camera_device": "Nikon D90",
            "resolution": "4288x2848",
            "aspect_ratio": "3:2",
            "compression": "JPEG Q95",
            "scene_category": "outdoor_landscape",
            "split": "test",
            "group_id": "raise_seq_102"
        },
        {
            "id": "real_cam_002",
            "path": "test/real/real_cam_002.jpg",
            "label": "REAL",
            "source": "Dresden",
            "generator": None,
            "generator_family": None,
            "camera_device": "Canon EOS 5D Mark II",
            "resolution": "5616x3744",
            "aspect_ratio": "3:2",
            "compression": "JPEG Q90",
            "scene_category": "architecture",
            "split": "test",
            "group_id": "dresden_seq_045"
        },
        {
            "id": "real_phone_001",
            "path": "test/real/real_phone_001.jpg",
            "label": "REAL",
            "source": "Smartphone_Pool",
            "generator": None,
            "generator_family": None,
            "camera_device": "iPhone 14 Pro",
            "resolution": "4032x3024",
            "aspect_ratio": "4:3",
            "compression": "JPEG Q85",
            "scene_category": "portrait",
            "split": "test",
            "group_id": "phone_session_12"
        },
        {
            "id": "real_social_001",
            "path": "test/real/real_social_001.jpg",
            "label": "REAL",
            "source": "Social_Reencode",
            "generator": None,
            "generator_family": None,
            "camera_device": "Samsung Galaxy S23",
            "resolution": "1920x1080",
            "aspect_ratio": "16:9",
            "compression": "JPEG Q50",
            "scene_category": "indoor",
            "split": "test",
            "group_id": "social_post_89"
        },
        {
            "id": "real_lowlight_001",
            "path": "test/real/real_lowlight_001.jpg",
            "label": "REAL",
            "source": "LowLight_Capture",
            "generator": None,
            "generator_family": None,
            "camera_device": "Sony Alpha 7 IV",
            "resolution": "3840x2160",
            "aspect_ratio": "16:9",
            "compression": "JPEG Q88",
            "scene_category": "night_scene",
            "split": "test",
            "group_id": "night_walk_04"
        },

        # --- TEST SPLIT: SYNTHETIC GENERATORS (Seen & Unseen) ---
        {
            "id": "synth_sd14_001",
            "path": "test/synthetic/synth_sd14_001.jpg",
            "label": "SYNTHETIC",
            "source": "CIFAKE_Eval",
            "generator": "stable_diffusion_1.4",
            "generator_family": "latent_diffusion",
            "camera_device": None,
            "resolution": "512x512",
            "aspect_ratio": "1:1",
            "compression": "JPEG Q90",
            "scene_category": "object",
            "split": "test",
            "group_id": "cifake_batch_01"
        },
        {
            "id": "synth_sdxl_001",
            "path": "test/synthetic/synth_sdxl_001.jpg",
            "label": "SYNTHETIC",
            "source": "SDXL_Eval",
            "generator": "stable_diffusion_xl",
            "generator_family": "latent_diffusion",
            "camera_device": None,
            "resolution": "1024x1024",
            "aspect_ratio": "1:1",
            "compression": "JPEG Q90",
            "scene_category": "portrait",
            "split": "test",
            "group_id": "sdxl_prompt_44"
        },
        {
            "id": "synth_mj6_001",
            "path": "test/synthetic/synth_mj6_001.jpg",
            "label": "SYNTHETIC",
            "source": "Midjourney_v6_Pool",
            "generator": "midjourney_v6",
            "generator_family": "proprietary_diffusion",
            "camera_device": None,
            "resolution": "2048x1536",
            "aspect_ratio": "4:3",
            "compression": "JPEG Q85",
            "scene_category": "landscape",
            "split": "test",
            "group_id": "mj6_gallery_08"
        },
        {
            "id": "synth_flux1_001",
            "path": "test/synthetic/synth_flux1_001.jpg",
            "label": "SYNTHETIC",
            "source": "Flux1_Dev_Pool",
            "generator": "flux_1",
            "generator_family": "flow_matching",
            "camera_device": None,
            "resolution": "1920x1080",
            "aspect_ratio": "16:9",
            "compression": "JPEG Q88",
            "scene_category": "architecture",
            "split": "test",
            "group_id": "flux_gen_19"
        },
        {
            "id": "synth_dalle3_001",
            "path": "test/synthetic/synth_dalle3_001.jpg",
            "label": "SYNTHETIC",
            "source": "DALL_E_3_Pool",
            "generator": "dall_e_3",
            "generator_family": "cascade_diffusion",
            "camera_device": None,
            "resolution": "1792x1024",
            "aspect_ratio": "16:9",
            "compression": "JPEG Q80",
            "scene_category": "indoor",
            "split": "test",
            "group_id": "dalle3_gen_77"
        }
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2)
    print(f"Sample manifest template created at: {output_path} ({len(sample)} items)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python manifest_validator.py <path_to_manifest.json> [dataset_root_dir]")
        sys.exit(1)

    mpath = sys.argv[1]
    root = sys.argv[2] if len(sys.argv) > 2 else ""
    valid, errs, stats = validate_manifest(mpath, root)
    if valid:
        print("Manifest validation SUCCESSFUL (Zero data leakage verified).")
        print(json.dumps(stats, indent=2))
    else:
        print("Manifest validation FAILED with errors:")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
