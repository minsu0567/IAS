#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
DRAEM Pixel-level Anomaly Score 분포 히스토그램 시각화 (v3)

v1 대비 변경점:
- 샘플링 없이 전체 픽셀을 사용하여 정확한 분포 시각화

사용법:
  python visualize_score_histogram_draem_v3.py \
    --checkpoint_rec checkpoints/rec_best.pth \
    --checkpoint_seg checkpoints/seg_best.pth \
    --dataset_path AD_dataset \
    --defect_types mounting_plate \
    --output_dir output/score_histogram_draem_v3
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import glob
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from models.draem_networks import ReconstructiveSubNetwork, DiscriminativeSubNetwork


# ─── Dataset ──────────────────────────────────────────────
class DRAEMTestDataset(Dataset):
    def __init__(self, dataset_path, defect_types, resize_shape=[256, 256]):
        self.resize_shape = resize_shape
        self.samples = []

        for dt in defect_types:
            good_dir = os.path.join(dataset_path, dt, 'test', 'good')
            for img_path in sorted(glob.glob(os.path.join(good_dir, '*.png'))):
                self.samples.append((img_path, None, 0))

            defect_dir = os.path.join(dataset_path, dt, 'test', 'defect')
            gt_dir = os.path.join(dataset_path, dt, 'ground_truth')
            for img_path in sorted(glob.glob(os.path.join(defect_dir, '*.png'))):
                fname = os.path.basename(img_path)
                idx_str = fname.split('.')[0]
                mask_path = os.path.join(gt_dir, 'defect', f"{idx_str}_mask.png")
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(gt_dir, f"{idx_str}_mask.png")
                self.samples.append((img_path, mask_path, 1))

        n_good = sum(1 for s in self.samples if s[2] == 0)
        n_defect = sum(1 for s in self.samples if s[2] == 1)
        print(f"DRAEMTestDataset: {len(self.samples)} samples "
              f"({n_good} good, {n_defect} defect)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, has_anomaly = self.samples[idx]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[0]))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        if mask_path is not None and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.resize_shape[1], self.resize_shape[0]))
            mask = mask.astype(np.float32) / 255.0
        else:
            mask = np.zeros((self.resize_shape[0], self.resize_shape[1]),
                            dtype=np.float32)

        mask = np.expand_dims(mask, axis=0)

        return {
            'image': image,
            'mask': mask,
            'has_anomaly': np.array([has_anomaly], dtype=np.float32)
        }


# ─── 히스토그램 시각화 ─────────────────────────────────────
def _draw_histogram(ax, normal_scores, anomaly_scores, bins, vmin, vmax,
                     ylim_max=None):
    ax.hist(normal_scores, bins=bins, range=(vmin, vmax),
            alpha=0.6, color='blue', label=f'Normal (n={len(normal_scores)})',
            density=True)
    ax.hist(anomaly_scores, bins=bins, range=(vmin, vmax),
            alpha=0.6, color='red', label=f'Anomaly (n={len(anomaly_scores)})',
            density=True)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    ax.tick_params(axis='x', which='major', length=6)
    ax.tick_params(axis='x', which='minor', length=3)
    if ylim_max is not None:
        ax.set_ylim(0, ylim_max)
    ax.set_xlabel('Anomaly Score', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.legend(fontsize=11)


def plot_histogram(normal_scores, anomaly_scores, save_path,
                   bins=200, xlim_min=None, xlim_max=None, ylim_max=None):
    all_scores = np.concatenate([normal_scores, anomaly_scores])
    vmin = xlim_min if xlim_min is not None else np.percentile(all_scores, 0.5)
    vmax = xlim_max if xlim_max is not None else np.percentile(all_scores, 99.5)

    # 전체 히스토그램 (linear scale)
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_histogram(ax, normal_scores, anomaly_scores, bins, vmin, vmax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    # log-scale 히스토그램 (FP 구간의 파란색 분포가 보임)
    fig, ax = plt.subplots(figsize=(10, 6))
    _draw_histogram(ax, normal_scores, anomaly_scores, bins, vmin, vmax)
    handles, labels = ax.get_legend_handles_labels()
    labels = [l.split(' (n=')[0] for l in labels]
    ax.legend(handles, labels, fontsize=11)
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1e3)
    plt.tight_layout()
    log_path = save_path.replace('.png', '_log.png')
    plt.savefig(log_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {log_path}")

    # 확대 히스토그램 (xlim, ylim 모두 지정된 경우)
    if xlim_min is not None and xlim_max is not None and ylim_max is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        _draw_histogram(ax, normal_scores, anomaly_scores, bins, vmin, vmax,
                         ylim_max=ylim_max)
        ax.set_xlim(xlim_min, xlim_max)
        plt.tight_layout()
        zoomed_path = save_path.replace('.png', '_zoomed.png')
        plt.savefig(zoomed_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {zoomed_path}")


# ─── Main ─────────────────────────────────────────────────
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description='DRAEM pixel-level anomaly score distribution histogram (v3, no sampling)')

    parser.add_argument('--checkpoint_rec', type=str, required=True,
                        help='Path to reconstructive model checkpoint')
    parser.add_argument('--checkpoint_seg', type=str, required=True,
                        help='Path to discriminative model checkpoint')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Dataset root path')
    parser.add_argument('--defect_types', nargs='+', type=str, default=['mounting_plate'],
                        help='Defect types to evaluate (e.g., mounting_plate)')
    parser.add_argument('--output_dir', type=str,
                        default='output/score_histogram_draem_v3',
                        help='Output directory')
    parser.add_argument('--bins', type=int, default=200,
                        help='Number of histogram bins')
    parser.add_argument('--xlim_min', type=float, default=None,
                        help='Histogram x-axis minimum (default: auto)')
    parser.add_argument('--xlim_max', type=float, default=None,
                        help='Histogram x-axis maximum (default: auto)')
    parser.add_argument('--ylim_max', type=float, default=None,
                        help='Histogram y-axis maximum (default: auto)')
    parser.add_argument('--gpu_id', type=int, default=0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ─── 모델 로드 ───
    print("Loading models...")
    model_rec = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model_rec.load_state_dict(
        torch.load(args.checkpoint_rec, map_location='cpu', weights_only=False))
    model_rec.cuda(args.gpu_id)
    model_rec.eval()

    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.load_state_dict(
        torch.load(args.checkpoint_seg, map_location='cpu', weights_only=False))
    model_seg.cuda(args.gpu_id)
    model_seg.eval()

    # ─── 데이터 로드 ───
    dataset = DRAEMTestDataset(args.dataset_path, args.defect_types)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # ─── 픽셀 score 수집 ───
    print("\nCollecting pixel scores...")
    normal_scores_all = []
    anomaly_scores_all = []

    for sample in dataloader:
        image = sample['image'].cuda(args.gpu_id)
        mask = sample['mask'].numpy()[0, 0]  # (H, W)
        is_anomaly = sample['has_anomaly'].numpy()[0, 0]

        rec_output = model_rec(image)
        joined_in = torch.cat((rec_output, image), dim=1)
        out_mask = model_seg(joined_in)
        out_mask_sm = torch.softmax(out_mask, dim=1)
        anomaly_map = out_mask_sm[0, 1, :, :].cpu().numpy()

        if is_anomaly == 0:
            normal_scores_all.append(anomaly_map.ravel())
        else:
            gt_binary = (mask > 0.5).astype(bool)
            normal_scores_all.append(anomaly_map[~gt_binary].ravel())
            anomaly_scores_all.append(anomaly_map[gt_binary].ravel())

    # ─── 검증 ───
    if not normal_scores_all:
        raise RuntimeError("No normal pixel scores collected. "
                           "Check that test/good/ exists and contains images.")
    if not anomaly_scores_all:
        raise RuntimeError("No anomaly pixel scores collected. "
                           "Check that test/defect/ exists and GT masks are found.")

    normal_scores = np.concatenate(normal_scores_all)
    anomaly_scores = np.concatenate(anomaly_scores_all)

    print(f"\n  Total normal pixels:  {len(normal_scores)}")
    print(f"  Total anomaly pixels: {len(anomaly_scores)}")

    # ─── 히스토그램 시각화 (전체 픽셀 사용) ───
    print("\nPlotting histogram...")
    defect_str = '_'.join(args.defect_types)
    plot_histogram(
        normal_scores, anomaly_scores,
        save_path=os.path.join(args.output_dir, 'score_histogram.png'),
        bins=args.bins,
        xlim_min=args.xlim_min,
        xlim_max=args.xlim_max,
        ylim_max=args.ylim_max,
)

    # ─── 통계 요약 출력 및 JSON 저장 ───
    stats = {
        'normal': {
            'count': int(len(normal_scores)),
            'mean': float(np.mean(normal_scores)),
            'std': float(np.std(normal_scores)),
            'median': float(np.median(normal_scores)),
        },
        'anomaly': {
            'count': int(len(anomaly_scores)),
            'mean': float(np.mean(anomaly_scores)),
            'std': float(np.std(anomaly_scores)),
            'median': float(np.median(anomaly_scores)),
        },
    }

    print("\n" + "=" * 50)
    print("  SCORE STATISTICS")
    print("=" * 50)
    print(f"  Normal  - mean: {stats['normal']['mean']:.4f}, "
          f"std: {stats['normal']['std']:.4f}, "
          f"median: {stats['normal']['median']:.4f}")
    print(f"  Anomaly - mean: {stats['anomaly']['mean']:.4f}, "
          f"std: {stats['anomaly']['std']:.4f}, "
          f"median: {stats['anomaly']['median']:.4f}")
    print("=" * 50)

    stats_path = os.path.join(args.output_dir, 'score_statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {stats_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
