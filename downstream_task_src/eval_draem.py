import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import cv2
import json
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from scipy.ndimage import label

from models.draem_networks import ReconstructiveSubNetwork, DiscriminativeSubNetwork


# ──────────────────────────────────────────────
# F1-max
# ──────────────────────────────────────────────
def compute_f1_max(gt, scores):
    """precision-recall curve 기반 F1-max 계산."""
    precisions, recalls, _ = precision_recall_curve(gt, scores)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    f1_max = np.max(f1_scores[np.isfinite(f1_scores)])
    return float(f1_max)


# ──────────────────────────────────────────────
# PRO (Per-Region Overlap)
# ──────────────────────────────────────────────
class _GroundTruthComponent:
    def __init__(self, anomaly_scores):
        self.anomaly_scores = sorted(anomaly_scores.tolist())
        self.index = 0

    def compute_overlap(self, threshold):
        while self.index < len(self.anomaly_scores) and self.anomaly_scores[self.index] <= threshold:
            self.index += 1
        return 1.0 - self.index / len(self.anomaly_scores)


def _collect_anomaly_scores(anomaly_maps, gt_maps):
    structure = np.ones((3, 3), dtype=int)
    components = []
    ok_scores = np.zeros(len(gt_maps) * gt_maps[0].size)
    ok_idx = 0
    for gt, pred in zip(gt_maps, anomaly_maps):
        labeled, n = label(gt, structure)
        num_ok = int((labeled == 0).sum())
        ok_scores[ok_idx:ok_idx + num_ok] = pred[labeled == 0]
        ok_idx += num_ok
        for k in range(n):
            components.append(_GroundTruthComponent(pred[labeled == (k + 1)]))
    ok_scores = np.sort(ok_scores[:ok_idx])
    return components, ok_scores


def compute_au_pro(gt_maps, pred_maps, integration_limit=0.3, num_thresholds=100):
    """AU-PRO 계산. gt_maps/pred_maps: list of 2D numpy arrays."""
    components, ok_scores = _collect_anomaly_scores(pred_maps, gt_maps)
    if len(components) == 0:
        return 0.0

    positions = np.linspace(0, len(ok_scores) - 1, num=num_thresholds, dtype=int)
    fprs, pros = [1.0], [1.0]
    for pos in positions:
        threshold = ok_scores[pos]
        fpr = 1.0 - (pos + 1) / len(ok_scores)
        pro = sum(c.compute_overlap(threshold) for c in components) / len(components)
        fprs.append(fpr)
        pros.append(pro)

    fprs = fprs[::-1]
    pros = pros[::-1]
    fprs = np.array(fprs)
    pros = np.array(pros)

    # trapezoidal integration up to integration_limit
    mask = fprs <= integration_limit
    if mask.sum() < 2:
        return 0.0
    x, y = fprs[mask], pros[mask]
    au_pro = float(np.trapz(y, x) / integration_limit)
    return au_pro


class DRAEMTestDataset(Dataset):
    """
    test/good (label=0, mask=zeros) + test/defect (label=1, ground_truth mask) 로딩.
    평가 시 실제 데이터를 사용.
    """
    def __init__(self, dataset_path, defect_types, resize_shape=[256, 256]):
        self.resize_shape = resize_shape
        self.samples = []  # (image_path, mask_path_or_None, has_anomaly)

        for dt in defect_types:
            # Good test images
            good_dir = os.path.join(dataset_path, dt, 'test', 'good')
            for img_path in sorted(glob.glob(os.path.join(good_dir, '*.png'))):
                self.samples.append((img_path, None, 0))

            # Defect test images
            defect_dir = os.path.join(dataset_path, dt, 'test', 'defect')
            gt_dir = os.path.join(dataset_path, dt, 'ground_truth')
            for img_path in sorted(glob.glob(os.path.join(defect_dir, '*.png'))):
                fname = os.path.basename(img_path)
                idx_str = fname.split('.')[0]
                mask_path = os.path.join(gt_dir, f"{idx_str}_mask.png")
                self.samples.append((img_path, mask_path, 1))

        print(f"DRAEMTestDataset: {len(self.samples)} samples ({sum(1 for s in self.samples if s[2]==0)} good, {sum(1 for s in self.samples if s[2]==1)} defect)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, has_anomaly = self.samples[idx]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[0]))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.resize_shape[1], self.resize_shape[0]))
            mask = mask.astype(np.float32) / 255.0
        else:
            mask = np.zeros((self.resize_shape[0], self.resize_shape[1]), dtype=np.float32)

        mask = np.expand_dims(mask, axis=0)

        return {
            'image': image,
            'mask': mask,
            'has_anomaly': np.array([has_anomaly], dtype=np.float32)
        }


@torch.no_grad()
def evaluate(args):
    img_dim = 256

    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load(args.checkpoint_rec, map_location='cuda:0'))
    model.cuda()
    model.eval()

    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.load_state_dict(torch.load(args.checkpoint_seg, map_location='cuda:0'))
    model_seg.cuda()
    model_seg.eval()

    dataset = DRAEMTestDataset(args.dataset_path, args.defect_types)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # anomaly map / heatmap 저장 디렉토리
    anomaly_map_dir = os.path.join(args.checkpoint_path, "anomaly_maps")
    heatmap_dir = os.path.join(args.checkpoint_path, "heatmaps")
    os.makedirs(anomaly_map_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)

    total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    mask_cnt = 0

    anomaly_score_gt = []
    anomaly_score_prediction = []

    # PRO 계산을 위해 2D 맵을 별도 저장
    gt_maps_2d = []
    pred_maps_2d = []

    for i_batch, sample_batched in enumerate(dataloader):
        gray_batch = sample_batched["image"].cuda()
        is_anomaly = sample_batched["has_anomaly"].numpy()[0, 0]
        anomaly_score_gt.append(is_anomaly)
        true_mask = sample_batched["mask"]
        true_mask_cv = true_mask.numpy()[0, :, :, :].transpose((1, 2, 0))

        gray_rec = model(gray_batch)
        joined_in = torch.cat((gray_rec.detach(), gray_batch), dim=1)

        out_mask = model_seg(joined_in)
        out_mask_sm = torch.softmax(out_mask, dim=1)

        out_mask_cv = out_mask_sm[0, 1, :, :].cpu().numpy()

        out_mask_averaged = F.avg_pool2d(
            out_mask_sm[:, 1:, :, :], 21, stride=1, padding=21 // 2
        ).cpu().numpy()
        image_score = np.max(out_mask_averaged)
        anomaly_score_prediction.append(image_score)

        flat_true_mask = true_mask_cv.flatten()
        flat_out_mask = out_mask_cv.flatten()
        total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
        total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
        mask_cnt += 1

        # PRO용 2D 맵 저장
        gt_maps_2d.append((true_mask_cv[:, :, 0] > 0.5).astype(np.uint8))
        pred_maps_2d.append(out_mask_cv)

        # ── Anomaly map 저장 (grayscale, 0~255) ──
        anomaly_map_uint8 = (out_mask_cv * 255).astype(np.uint8)
        img_path = dataset.samples[i_batch][0]
        fname = os.path.splitext(os.path.basename(img_path))[0]
        sub_dir = "defect" if is_anomaly == 1 else "good"
        os.makedirs(os.path.join(anomaly_map_dir, sub_dir), exist_ok=True)
        os.makedirs(os.path.join(heatmap_dir, sub_dir), exist_ok=True)
        cv2.imwrite(os.path.join(anomaly_map_dir, sub_dir, f"{fname}.png"), anomaly_map_uint8)

        # ── Heatmap 저장 (원본 이미지 위에 anomaly map 오버레이) ──
        heatmap_color = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_JET)
        orig_img = (gray_batch[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        orig_img_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(orig_img_bgr, 0.5, heatmap_color, 0.5, 0)
        cv2.imwrite(os.path.join(heatmap_dir, sub_dir, f"{fname}.png"), overlay)

    anomaly_score_prediction = np.array(anomaly_score_prediction)
    anomaly_score_gt = np.array(anomaly_score_gt)

    # ── Image-level metrics ──
    image_auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
    image_ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)
    image_f1_max = compute_f1_max(anomaly_score_gt, anomaly_score_prediction)

    # ── Pixel-level metrics ──
    total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt].astype(np.uint8)
    total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
    pixel_auroc = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
    pixel_ap = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
    pixel_f1_max = compute_f1_max(total_gt_pixel_scores, total_pixel_scores)

    # ── PRO ──
    pro = compute_au_pro(gt_maps_2d, pred_maps_2d)

    # ── Print ──
    print("==============================")
    print(f"Image AUROC:  {image_auroc:.4f}")
    print(f"Image AP:     {image_ap:.4f}")
    print(f"Image F1-max: {image_f1_max:.4f}")
    print(f"Pixel AUROC:  {pixel_auroc:.4f}")
    print(f"Pixel AP:     {pixel_ap:.4f}")
    print(f"Pixel F1-max: {pixel_f1_max:.4f}")
    print(f"PRO (AU-PRO): {pro:.4f}")
    print("==============================")

    # ── Save to JSON ──
    results = {
        "image_auroc": round(image_auroc, 4),
        "image_ap": round(image_ap, 4),
        "image_f1_max": round(image_f1_max, 4),
        "pixel_auroc": round(pixel_auroc, 4),
        "pixel_ap": round(pixel_ap, 4),
        "pixel_f1_max": round(pixel_f1_max, 4),
        "pro": round(pro, 4),
    }
    out_path = os.path.join(args.checkpoint_path, "eval_results.json")
    os.makedirs(args.checkpoint_path, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")
    print(f"Anomaly maps saved to {anomaly_map_dir}")
    print(f"Heatmaps saved to {heatmap_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Dataset root path (e.g., AD_dataset_anogen)')
    parser.add_argument('--defect_types', nargs='+', type=str, default=['mounting_plate'],
                        help='Defect types to evaluate (e.g., mounting_plate)')
    parser.add_argument('--checkpoint_rec', type=str, required=True,
                        help='Path to reconstructive model checkpoint')
    parser.add_argument('--checkpoint_seg', type=str, required=True,
                        help='Path to discriminative model checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints_draem/',
                        help='Directory to save eval_results.json')
    parser.add_argument('--gpu_id', type=int, default=0)

    args = parser.parse_args()

    with torch.cuda.device(args.gpu_id):
        evaluate(args)
