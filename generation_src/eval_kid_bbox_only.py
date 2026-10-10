"""
KID evaluation with bounding-box cropping.

Real images are cropped using per-image bounding boxes.
Generated images are cropped using a single shared bounding box.
Bounding box coordinates are normalized (center_x, center_y, width, height) in [0, 1].

Usage:
    python eval_kid_bbox_only.py \
        --dataset_root AD_dataset_s_ROI \
        --generated_root generated_output \
        --defect_type mounting_plate \
        --bbox_guide KID_guide.txt
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ============================================================
# Bounding Box Parsing
# ============================================================

def parse_bbox_guide(guide_path):
    """Parse KID_guide.txt and return per-image bbox dict and generated bbox.

    Returns:
        real_bboxes: dict {filename: (cx, cy, w, h)}
        gen_bbox: (cx, cy, w, h)
    """
    real_bboxes = {}
    gen_bbox = None

    with open(guide_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "dataset_root" in line:
            section = "real"
            continue
        if "generated_root" in line:
            section = "gen"
            continue
        if line.startswith("바운딩박스") or line.startswith("0~1"):
            continue

        if section == "real":
            match = re.match(r'(\S+\.png)\s*:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
            if match:
                fname = match.group(1)
                cx, cy, w, h = float(match.group(2)), float(match.group(3)), float(match.group(4)), float(match.group(5))
                real_bboxes[fname] = (cx, cy, w, h)
        elif section == "gen":
            parts = line.split()
            if len(parts) == 4:
                try:
                    gen_bbox = tuple(float(v) for v in parts)
                except ValueError:
                    pass

    return real_bboxes, gen_bbox


def crop_bbox(img, cx, cy, w, h):
    """Crop image using normalized (center_x, center_y, width, height) coordinates."""
    img_w, img_h = img.size
    x1 = max(0, int((cx - w / 2) * img_w))
    y1 = max(0, int((cy - h / 2) * img_h))
    x2 = min(img_w, int((cx + w / 2) * img_w))
    y2 = min(img_h, int((cy + h / 2) * img_h))
    return img.crop((x1, y1, x2, y2))


# ============================================================
# Inception Feature Extraction
# ============================================================

INCEPTION_URL = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'


def get_inception_model(device):
    """Download (with caching) and load the NVIDIA Inception model."""
    cache_dir = os.path.join(torch.hub.get_dir(), 'checkpoints')
    os.makedirs(cache_dir, exist_ok=True)
    url_md5 = hashlib.md5(INCEPTION_URL.encode('utf-8')).hexdigest()
    cache_file = os.path.join(cache_dir, f'inception-2015-12-05-{url_md5[:8]}.pt')
    if not os.path.exists(cache_file):
        print(f"Downloading Inception model to {cache_file} ...")
        torch.hub.download_url_to_file(INCEPTION_URL, cache_file)
    model = torch.jit.load(cache_file).eval().to(device)
    return model


def extract_features_with_bbox(image_paths, bboxes, detector, device, batch_size=32):
    """Extract Inception features from cropped images.

    Args:
        image_paths: list of image paths
        bboxes: either a dict {filename: (cx,cy,w,h)} for per-image bbox,
                or a single tuple (cx,cy,w,h) for shared bbox
        detector: Inception model
        device: torch device
        batch_size: batch size

    Returns:
        np.ndarray [N, 2048]
    """
    all_features = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting Inception features"):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []
        for p in batch_paths:
            img = Image.open(p).convert('RGB')

            if isinstance(bboxes, dict):
                fname = os.path.basename(p)
                if fname in bboxes:
                    cx, cy, w, h = bboxes[fname]
                    img = crop_bbox(img, cx, cy, w, h)
                else:
                    print(f"  Warning: no bbox for {fname}, using full image")
            else:
                cx, cy, w, h = bboxes
                img = crop_bbox(img, cx, cy, w, h)

            img = img.resize((299, 299), Image.LANCZOS)
            tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)
            batch_tensors.append(tensor)

        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            features = detector(batch, return_features=True)
        all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


# ============================================================
# KID Computation
# ============================================================

def compute_kid(real_features, gen_features, num_subsets=100, max_subset_size=1000):
    """Compute KID with mean and std across subsets."""
    n = real_features.shape[1]
    m = min(real_features.shape[0], gen_features.shape[0], max_subset_size)

    scores = []
    for _ in range(num_subsets):
        x = gen_features[np.random.choice(gen_features.shape[0], m, replace=(gen_features.shape[0] < m))]
        y = real_features[np.random.choice(real_features.shape[0], m, replace=(real_features.shape[0] < m))]

        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3

        kid_subset = ((a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m) / m
        scores.append(kid_subset)

    scores = np.array(scores)
    return float(scores.mean()), float(scores.std())


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="KID evaluation with bounding-box cropping")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to AD_dataset_s_ROI")
    parser.add_argument("--generated_root", type=str, required=True,
                        help="Path to generated output (cycle_*/step_*/image_*/generated.png)")
    parser.add_argument("--defect_type", type=str, required=True,
                        help="Defect type to evaluate (e.g., 'mounting_plate')")
    parser.add_argument("--bbox_guide", type=str, required=True,
                        help="Path to KID_guide.txt with bounding box coordinates")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_subsets", type=int, default=100)
    parser.add_argument("--max_subset_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON (default: {generated_root}/eval_kid_only.json)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse bbox guide
    real_bboxes, gen_bbox = parse_bbox_guide(args.bbox_guide)
    print(f"Loaded {len(real_bboxes)} real bboxes, gen bbox: {gen_bbox}")

    # Collect image paths
    real_dir = os.path.join(args.dataset_root, args.defect_type, "train")
    real_paths = sorted(glob.glob(os.path.join(real_dir, "*.png")))

    gen_paths = sorted(glob.glob(
        os.path.join(args.generated_root, "cycle_*", "step_*", "image_*", "generated.png")
    ))

    print(f"Defect type: {args.defect_type}")
    print(f"Real images: {len(real_paths)} (from {real_dir})")
    print(f"Generated images: {len(gen_paths)} (from {args.generated_root})")

    if len(real_paths) == 0:
        raise FileNotFoundError(f"No real images found in {real_dir}")
    if len(gen_paths) == 0:
        raise FileNotFoundError(f"No generated images found in {args.generated_root}")

    # Extract features with bbox cropping
    print("\n" + "=" * 60)
    print("Extracting features (bbox-cropped)...")
    print("=" * 60)

    detector = get_inception_model(device)

    real_features = extract_features_with_bbox(real_paths, real_bboxes, detector, device, args.batch_size)
    gen_features = extract_features_with_bbox(gen_paths, gen_bbox, detector, device, args.batch_size)

    print(f"Real features shape: {real_features.shape}")
    print(f"Generated features shape: {gen_features.shape}")

    del detector
    torch.cuda.empty_cache()

    # Compute KID
    print("\n" + "=" * 60)
    print("Computing KID...")
    print("=" * 60)

    kid_mean, kid_std = compute_kid(real_features, gen_features,
                                    num_subsets=args.num_subsets,
                                    max_subset_size=args.max_subset_size)

    # Results
    print(f"\n{'=' * 60}")
    print(f"Results (defect_type={args.defect_type})")
    print(f"{'=' * 60}")
    print(f"  KID: {kid_mean:.6f} +/- {kid_std:.6f}")
    print(f"{'=' * 60}")

    # Save results
    output_path = args.output if args.output else os.path.join(args.generated_root, "eval_kid_only.json")
    results = {
        "defect_type": args.defect_type,
        "num_real_images": len(real_paths),
        "num_gen_images": len(gen_paths),
        "KID_mean": kid_mean,
        "KID_std": kid_std,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
