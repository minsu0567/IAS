#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
PatchCore Pixel-level Anomaly Score 분포 히스토그램 시각화 (v3)

- 샘플링 없이 전체 픽셀을 사용하여 정확한 분포 시각화
- WideResNet50 backbone 고정

사용법:
  python bin/visualize_score_histogram_v3.py \
    --data_path AD_dataset_application3 \
    --defect_type mounting_plate \
    --output_dir output/score_histogram_patchcore_v3
"""

import json
import logging
import os
import sys

import numpy as np
import torch
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─── patchcore 패키지 경로 추가 ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_DIR, 'src')
sys.path.insert(0, SRC_DIR)

import patchcore.backbones
import patchcore.common
import patchcore.patchcore
import patchcore.sampler

LOGGER = logging.getLogger(__name__)


# ─── Argument Parser ──────────────────────────────────────
def get_argparse():
    parser = argparse.ArgumentParser(
        description='PatchCore pixel-level anomaly score distribution histogram (v3)')

    # 데이터셋
    parser.add_argument('--data_path', type=str, required=True,
                        help='Dataset root path')
    parser.add_argument('--defect_type', type=str, required=True,
                        help='Defect type / category name (e.g., mounting_plate)')

    # 출력
    parser.add_argument('--output_dir', type=str,
                        default='output/score_histogram_patchcore_v3',
                        help='Output directory')

    # PatchCore 설정
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--imagesize', type=int, default=224)
    parser.add_argument('--f_coreset', type=float, default=0.1,
                        help='Coreset subsampling fraction')
    parser.add_argument('--layers', type=str, nargs='+',
                        default=['layer2', 'layer3'],
                        help='Layers to extract features from (e.g., --layers layer2 layer3)')
    parser.add_argument('--seed', type=int, default=0)

    # 히스토그램 설정
    parser.add_argument('--bins', type=int, default=200,
                        help='Number of histogram bins')
    parser.add_argument('--xlim_min', type=float, default=None,
                        help='Histogram x-axis minimum (default: auto)')
    parser.add_argument('--xlim_max', type=float, default=None,
                        help='Histogram x-axis maximum (default: auto)')
    parser.add_argument('--ylim_max', type=float, default=None,
                        help='Histogram y-axis maximum (default: auto)')

    return parser.parse_args()


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
def main():
    logging.basicConfig(level=logging.INFO)
    args = get_argparse()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ─── Seed 고정 ───
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ─── CustomDataset 로드 ───
    print("\nLoading dataset...")
    from patchcore.datasets.custom import CustomDataset
    from patchcore.datasets.mvtec import DatasetSplit

    train_dataset = CustomDataset(
        args.data_path,
        classname=args.defect_type,
        resize=args.resize,
        imagesize=args.imagesize,
        split=DatasetSplit.TRAIN,
        seed=args.seed,
    )
    test_dataset = CustomDataset(
        args.data_path,
        classname=args.defect_type,
        resize=args.resize,
        imagesize=args.imagesize,
        split=DatasetSplit.TEST,
        seed=args.seed,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False,
        num_workers=4, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=2, shuffle=False,
        num_workers=4, pin_memory=True)

    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # ─── PatchCore 빌드 (WideResNet50 고정) ───
    print("\nBuilding PatchCore (WideResNet50)...")
    backbone = patchcore.backbones.load("wideresnet50")
    backbone.name = "wideresnet50"
    backbone.seed = None

    nn_method = patchcore.common.FaissNN(on_gpu=False, num_workers=4)
    sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(
        args.f_coreset, device)

    patch_core = patchcore.patchcore.PatchCore(device)
    patch_core.load(
        backbone=backbone,
        layers_to_extract_from=args.layers,
        device=device,
        input_shape=train_dataset.imagesize,
        pretrain_embed_dimension=512,
        target_embed_dimension=512,
        patchsize=3,
        featuresampler=sampler,
        anomaly_scorer_num_nn=1,
        nn_method=nn_method,
    )

    # ─── 학습 ───
    print("\nTraining PatchCore...")
    torch.cuda.empty_cache()
    patch_core.fit(train_dataloader)

    # ─── 추론 ───
    print("\nPredicting on test data...")
    torch.cuda.empty_cache()
    scores, segmentations, labels_gt, masks_gt = patch_core.predict(test_dataloader)

    # ─── Segmentation map 정규화 (run_patchcore_custom.py 동일) ───
    segmentations = np.array(segmentations)
    min_scores = (
        segmentations.reshape(len(segmentations), -1)
        .min(axis=-1)
        .reshape(-1, 1, 1)
    )
    max_scores = (
        segmentations.reshape(len(segmentations), -1)
        .max(axis=-1)
        .reshape(-1, 1, 1)
    )
    segmentations = (segmentations - min_scores) / (max_scores - min_scores + 1e-8)

    # ─── GT mask 기반 픽셀 분리 ───
    print("\nCollecting pixel scores...")
    normal_scores_all = []
    anomaly_scores_all = []

    for i in range(len(segmentations)):
        seg_map = segmentations[i]   # (H, W)
        mask = np.array(masks_gt[i]) # mask from dataloader

        # 2D로 변환
        while mask.ndim > 2:
            mask = mask[0]

        # segmentation map 크기에 맞게 mask 리사이즈
        if mask.shape != seg_map.shape:
            mask_tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor, size=seg_map.shape, mode='nearest')
            mask = mask_tensor[0, 0].numpy()

        gt_binary = (mask > 0.5).astype(bool)

        if labels_gt[i] == 0:
            # good 이미지: 모든 픽셀 → normal
            normal_scores_all.append(seg_map.ravel())
        else:
            # defect 이미지: GT 기반 분리
            normal_scores_all.append(seg_map[~gt_binary].ravel())
            anomaly_scores_all.append(seg_map[gt_binary].ravel())

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

    # ─── 히스토그램 시각화 ───
    print("\nPlotting histogram...")
    plot_histogram(
        normal_scores, anomaly_scores,
        save_path=os.path.join(args.output_dir, 'score_histogram.png'),
        bins=args.bins,
        xlim_min=args.xlim_min,
        xlim_max=args.xlim_max,
        ylim_max=args.ylim_max)

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
