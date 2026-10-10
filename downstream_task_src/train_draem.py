import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
import numpy as np
import os
import glob
import cv2
import random
import argparse
from tqdm import tqdm

from models.draem_networks import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from models.draem_loss import FocalLoss, SSIM, DiceLoss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class PairedTrainDataset(Dataset):
    """
    train/good, train/defect, inpainting_mask 를 1:1 paired 로 로딩.
    50% 확률로 정상(good만) / 결함(good+defect+mask) 반환.
    """
    def __init__(self, dataset_path, defect_types, resize_shape=[256, 256]):
        self.resize_shape = resize_shape
        self.samples = []  # (good_path, defect_path, mask_path)

        for dt in defect_types:
            good_dir = os.path.join(dataset_path, dt, 'train', 'good')
            defect_dir = os.path.join(dataset_path, dt, 'train', 'defect')
            mask_dir = os.path.join(dataset_path, dt, 'inpainting_mask')
            good_images = sorted(glob.glob(os.path.join(good_dir, '*.png')))
            for good_path in good_images:
                fname = os.path.basename(good_path)
                idx_str = fname.split('.')[0]
                defect_path = os.path.join(defect_dir, fname)
                mask_path = os.path.join(mask_dir, f"{idx_str}_mask.png")
                self.samples.append((good_path, defect_path, mask_path))

        print(f"PairedTrainDataset: {len(self.samples)} samples loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        good_path, defect_path, mask_path = self.samples[idx]

        # 50% 확률로 정상 샘플 반환
        if torch.rand(1).item() > 0.5:
            image = cv2.imread(good_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[0]))
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            mask = np.zeros((1, self.resize_shape[0], self.resize_shape[1]), dtype=np.float32)
            return {
                'image': image,
                'augmented_image': image.copy(),
                'anomaly_mask': mask,
                'has_anomaly': np.array([0.0], dtype=np.float32)
            }
        else:
            image = cv2.imread(good_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (self.resize_shape[1], self.resize_shape[0]))
            image = image.astype(np.float32) / 255.0

            aug_image = cv2.imread(defect_path, cv2.IMREAD_COLOR)
            aug_image = cv2.resize(aug_image, (self.resize_shape[1], self.resize_shape[0]))
            aug_image = aug_image.astype(np.float32) / 255.0

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.resize_shape[1], self.resize_shape[0]))
            mask = mask.astype(np.float32) / 255.0
            mask = np.expand_dims(mask, axis=0)

            image = np.transpose(image, (2, 0, 1))
            aug_image = np.transpose(aug_image, (2, 0, 1))
            return {
                'image': image,
                'augmented_image': aug_image,
                'anomaly_mask': mask,
                'has_anomaly': np.array([1.0], dtype=np.float32)
            }


def train(args):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    model.cuda()
    model.apply(weights_init)

    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.cuda()
    model_seg.apply(weights_init)

    optimizer = torch.optim.Adam([
        {"params": model.parameters(), "lr": args.lr},
        {"params": model_seg.parameters(), "lr": args.lr}
    ])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [400, 600], gamma=0.1, last_epoch=-1)

    loss_l2 = torch.nn.MSELoss()
    loss_ssim = SSIM()
    loss_seg_fn = DiceLoss() if args.loss_seg == 'dice' else FocalLoss()

    dataset = PairedTrainDataset(args.dataset_path, args.defect_types)
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=2, pin_memory=True)

    for epoch in tqdm(range(args.epochs)):
        model.train()
        model_seg.train()

        for i_batch, sample_batched in enumerate(dataloader):
            gray_batch = sample_batched["image"].cuda()
            aug_gray_batch = sample_batched["augmented_image"].cuda()
            anomaly_mask = sample_batched["anomaly_mask"].cuda()

            gray_rec = model(aug_gray_batch)
            joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

            out_mask = model_seg(joined_in)
            out_mask_sm = torch.softmax(out_mask, dim=1)

            l2_loss = loss_l2(gray_rec, gray_batch)
            ssim_loss = loss_ssim(gray_rec, gray_batch)
            segment_loss = loss_seg_fn(out_mask_sm, anomaly_mask)
            if segment_loss.dim() > 0:
                segment_loss = segment_loss.mean()

            loss = l2_loss + ssim_loss + segment_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch}: l2={l2_loss.item():.4f}, ssim={ssim_loss.item():.4f}, seg={segment_loss.item():.4f}")
            torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f"draem_rec_{epoch}.pckl"))
            torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, f"draem_seg_{epoch}.pckl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Dataset root path (e.g., AD_dataset_anogen)')
    parser.add_argument('--defect_types', nargs='+', type=str, default=['mounting_plate'],
                        help='Defect types to use (e.g., mounting_plate)')
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints_draem/')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--loss_seg', type=str, default='focal', choices=['focal', 'dice'])

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    with torch.cuda.device(args.gpu_id):
        train(args)
