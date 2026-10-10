#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Pixel-level Anomaly Score 분포 히스토그램 시각화 (v3)

v1 대비 변경점:
- 샘플링 없이 전체 픽셀을 사용하여 정확한 분포 시각화

사용법:
  python visualize_score_histogram_v3.py \
    --teacher output/trainings/1/teacher_final.pth \
    --student output/trainings/1/student_final.pth \
    --autoencoder output/trainings/1/autoencoder_final.pth \
    --dataset_path AD_dataset_application3 \
    --category 1 \
    --output_dir output/score_histogram_v3
"""

import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import argparse
import os
import sys
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ─── 환경 감지 ─────────────────────────────────────────────
def detect_environment():
    try:
        import google.colab
        return 'colab'
    except ImportError:
        return 'local'


ENVIRONMENT = detect_environment()

if ENVIRONMENT == 'colab':
    EFFICIENTAD_PATH = '/content/EfficientAD'
else:
    EFFICIENTAD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, EFFICIENTAD_PATH)

from common import (get_autoencoder, get_pdn_small, get_pdn_medium,
                    ImageFolderWithoutTarget, ImageFolderWithPath,
                    InfiniteDataloader)


# ─── Constants ─────────────────────────────────────────────
SEED = 42
IMAGE_SIZE = 256
OUT_CHANNELS = 384


# ─── Argument Parser ──────────────────────────────────────
def get_argparse():
    parser = argparse.ArgumentParser(
        description='Pixel-level anomaly score distribution histogram (v3, no sampling)')

    # 모델 가중치
    parser.add_argument('--teacher', type=str, required=True,
                        help='Path to teacher_final.pth')
    parser.add_argument('--student', type=str, required=True,
                        help='Path to student_final.pth')
    parser.add_argument('--autoencoder', type=str, required=True,
                        help='Path to autoencoder_final.pth')

    # 데이터셋
    parser.add_argument('-d', '--dataset_path', type=str, required=True,
                        help='Path to dataset root')
    parser.add_argument('-c', '--category', type=str, required=True,
                        help='Category to evaluate (e.g., 1)')

    # 출력
    parser.add_argument('-o', '--output_dir', type=str,
                        default='output/score_histogram_v3',
                        help='Output directory')

    # 설정
    parser.add_argument('--validation_ratio', type=float, default=0.2,
                        help='Validation ratio (must match training setting)')
    parser.add_argument('--bins', type=int, default=200,
                        help='Number of histogram bins')
    parser.add_argument('--xlim_min', type=float, default=None,
                        help='Histogram x-axis minimum (default: auto)')
    parser.add_argument('--xlim_max', type=float, default=None,
                        help='Histogram x-axis maximum (default: auto)')
    parser.add_argument('--ylim_max', type=float, default=None,
                        help='Histogram y-axis maximum (default: auto)')

    return parser.parse_args()


# ─── Transform ─────────────────────────────────────────────
def get_default_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# ─── Teacher Normalization ─────────────────────────────────
@torch.no_grad()
def teacher_normalization(teacher, train_loader, on_gpu):
    mean_outputs = []
    for train_image, _ in tqdm(train_loader, desc='Computing teacher mean'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output.cpu())

    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]
    if on_gpu:
        channel_mean = channel_mean.cuda()

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing teacher std'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance.cpu())

    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)
    if on_gpu:
        channel_std = channel_std.cuda()

    return channel_mean, channel_std


# ─── Predict ──────────────────────────────────────────────
@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)

    map_st = torch.mean(
        (teacher_output - student_output[:, :OUT_CHANNELS]) ** 2,
        dim=1, keepdim=True)
    map_ae = torch.mean(
        (autoencoder_output - student_output[:, OUT_CHANNELS:]) ** 2,
        dim=1, keepdim=True)

    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)

    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined


# ─── Map Normalization ────────────────────────────────────
@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, on_gpu):
    maps_st = []
    maps_ae = []

    for image, _ in tqdm(validation_loader, desc='Map normalization'):
        if on_gpu:
            image = image.cuda()

        teacher_output = teacher(image)
        teacher_output = (teacher_output - teacher_mean) / teacher_std
        student_output = student(image)
        autoencoder_output = autoencoder(image)

        map_st = torch.mean(
            (teacher_output - student_output[:, :OUT_CHANNELS]) ** 2,
            dim=1, keepdim=True)
        map_ae = torch.mean(
            (autoencoder_output - student_output[:, OUT_CHANNELS:]) ** 2,
            dim=1, keepdim=True)

        maps_st.append(map_st)
        maps_ae.append(map_ae)

    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)

    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)

    return q_st_start, q_st_end, q_ae_start, q_ae_end


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
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    config = get_argparse()
    os.makedirs(config.output_dir, exist_ok=True)

    on_gpu = torch.cuda.is_available()
    print(f"Environment: {ENVIRONMENT.upper()}")
    print(f"Device: {'GPU' if on_gpu else 'CPU'}")

    # ─── 모델 로드 ───
    print("\nLoading models...")
    teacher = torch.load(config.teacher, map_location='cpu', weights_only=False)
    student = torch.load(config.student, map_location='cpu', weights_only=False)
    autoencoder = torch.load(config.autoencoder, map_location='cpu',
                             weights_only=False)

    teacher.eval()
    student.eval()
    autoencoder.eval()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    # ─── 데이터 경로 ───
    category_path = os.path.join(config.dataset_path, config.category)
    test_dir = os.path.join(category_path, 'test')
    gt_dir = os.path.join(category_path, 'ground_truth')

    print(f"\nDataset:  {config.dataset_path}")
    print(f"Category: {config.category}")

    # ─── Training data 로드 (정규화 파라미터 계산용) ───
    default_transform = get_default_transform()
    train_transform_fn = lambda img: (default_transform(img),
                                      default_transform(img))

    full_train_set = ImageFolderWithoutTarget(
        os.path.join(category_path, 'train'),
        transform=transforms.Lambda(train_transform_fn))

    train_size = int((1.0 - config.validation_ratio) * len(full_train_set))
    validation_size = len(full_train_set) - train_size
    rng = torch.Generator().manual_seed(SEED)
    train_set, validation_set = torch.utils.data.random_split(
        full_train_set, [train_size, validation_size], rng)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=False,
                              num_workers=4, pin_memory=True)
    validation_loader = DataLoader(validation_set, batch_size=1)

    print(f"  Train: {train_size}, Validation: {validation_size}")

    # ─── Teacher Normalization ───
    print("\nComputing teacher normalization...")
    teacher_mean, teacher_std = teacher_normalization(
        teacher, train_loader, on_gpu)

    # ─── Map Normalization ───
    print("Computing map normalization...")
    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        validation_loader, teacher, student, autoencoder,
        teacher_mean, teacher_std, on_gpu)

    # ─── 테스트 이미지 순회 → 픽셀 score 수집 ───
    print("\nCollecting pixel scores from test images...")
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    normal_scores_all = []
    anomaly_scores_all = []

    for subdir in sorted(os.listdir(test_dir)):
        subdir_path = os.path.join(test_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        image_files = sorted([
            f for f in os.listdir(subdir_path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
        ])

        for img_file in tqdm(image_files, desc=f'  {subdir}'):
            img_path = os.path.join(subdir_path, img_file)
            img_name = os.path.splitext(img_file)[0]

            image = Image.open(img_path).convert('RGB')
            image_tensor = default_transform(image).unsqueeze(0)
            if on_gpu:
                image_tensor = image_tensor.cuda()

            with torch.no_grad():
                anomaly_map = predict(
                    image_tensor, teacher, student, autoencoder,
                    teacher_mean, teacher_std,
                    q_st_start, q_st_end, q_ae_start, q_ae_end)

            anomaly_map = torch.nn.functional.pad(anomaly_map, (4, 4, 4, 4))
            anomaly_map = torch.nn.functional.interpolate(
                anomaly_map, (image.height, image.width), mode='bilinear')
            anomaly_map = anomaly_map[0, 0].cpu().numpy()

            if subdir == 'good':
                normal_scores_all.append(anomaly_map.ravel())
            else:
                gt_mask_path = os.path.join(gt_dir, subdir, img_name + '_mask.png')
                if not os.path.exists(gt_mask_path):
                    gt_mask_path = os.path.join(gt_dir, img_name + '_mask.png')
                if not os.path.exists(gt_mask_path):
                    print(f"  WARNING: GT mask not found: {gt_mask_path}")
                    continue

                gt_mask = np.asarray(
                    Image.open(gt_mask_path).convert('L'))
                gt_binary = (gt_mask > 127).astype(bool)

                normal_scores_all.append(anomaly_map[~gt_binary].ravel())
                anomaly_scores_all.append(anomaly_map[gt_binary].ravel())

    # ─── Score 합치기 ───
    if not normal_scores_all:
        raise RuntimeError("No normal pixel scores collected. "
                           "Check that test/good/ directory exists and contains images.")
    if not anomaly_scores_all:
        raise RuntimeError("No anomaly pixel scores collected. "
                           "Check that test/defect/ directory exists and GT masks are found.\n"
                           "GT mask naming convention: {img_name}_mask.png under ground_truth/")

    normal_scores = np.concatenate(normal_scores_all)
    anomaly_scores = np.concatenate(anomaly_scores_all)

    print(f"\n  Total normal pixels:  {len(normal_scores)}")
    print(f"  Total anomaly pixels: {len(anomaly_scores)}")

    # ─── 히스토그램 시각화 (전체 픽셀 사용) ───
    print("\nPlotting histogram...")
    plot_histogram(
        normal_scores, anomaly_scores,
        save_path=os.path.join(config.output_dir, 'score_histogram.png'),
        bins=config.bins,
        xlim_min=config.xlim_min,
        xlim_max=config.xlim_max,
        ylim_max=config.ylim_max,
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

    stats_path = os.path.join(config.output_dir, 'score_statistics.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {stats_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
