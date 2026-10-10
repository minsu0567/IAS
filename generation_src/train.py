"""
Training script for Simplified Anomaly Synthesis Pipeline

Key features:
- NO RFA (Region-Focused Attention)
- NO IP-Adapter
- NO I_mask dependency for condition generation
- Simple visual+text concat approach (like anomalydiffusion-master)

Pipeline:
1. I_ref → CLIP Visual Encoder → Linear → P_a
2. prompt → CLIP Text Encoder → Linear → P_b
3. P_c = concat(P_a, P_b) → U-Net condition
4. Standard cross attention (no decoupled attention)
"""

import os
import argparse
import time
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
from peft import LoraConfig

from condition_encoder import ConditionEncoder


class CustomAnomalyDataset(Dataset):
    """
    Custom dataset for AD_dataset_s_ROI

    Simplified version - NO I_mask needed for condition generation
    Only uses:
    - I_ref (reference defect image)
    - expanded_mask (for loss weighting)
    - prompt (text description)
    """

    def __init__(self, data_root, defect_type, size=512):
        """
        Args:
            data_root: str - path to AD_dataset_s_ROI folder
            defect_type: str - 'mounting_plate'
            size: int - image size (default 512)
        """
        super().__init__()

        self.data_root = Path(data_root)
        self.defect_type = defect_type
        self.size = size

        # Set paths
        self.train_dir = self.data_root / defect_type / "train"  # I_ref
        self.expanded_mask_dir = self.data_root / defect_type / "ground_truth"  # For loss weighting
        self.prompt_file = self.data_root / defect_type / f"prompt_{defect_type}.txt"

        # Load prompt
        with open(self.prompt_file, 'r', encoding='utf-8') as f:
            self.prompt = f.read().strip()

        # Collect image and mask pairs
        self.data_pairs = []
        for img_file in sorted(self.train_dir.glob("*.png")):
            img_name = img_file.stem
            expanded_mask_file = self.expanded_mask_dir / f"{img_name}_mask.png"

            if expanded_mask_file.exists():
                self.data_pairs.append({
                    'image': img_file,
                    'expanded_mask': expanded_mask_file
                })

        print(f"Loaded {len(self.data_pairs)} training pairs for defect type '{defect_type}'")
        print(f"Prompt: {self.prompt}")

        # Image transforms for VAE input (normalize to [-1, 1])
        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # CLIP image processor for visual encoder
        self.clip_image_processor = CLIPImageProcessor()

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        pair = self.data_pairs[idx]

        # Load images
        image = Image.open(pair['image']).convert("RGB")
        expanded_mask = Image.open(pair['expanded_mask']).convert("L")

        # Process image for VAE (I_ref)
        image_tensor = self.transform(image)

        # Process image for CLIP visual encoder
        clip_image = self.clip_image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        # Process expanded mask for loss weighting (64x64 = latent size)
        expanded_mask_resized = expanded_mask.resize((64, 64))
        expanded_mask_tensor = torch.tensor(np.array(expanded_mask_resized), dtype=torch.float32) / 255.0
        expanded_mask_tensor = (expanded_mask_tensor > 0.5).float()

        return {
            "image": image_tensor,  # For VAE encoding
            "clip_image": clip_image,  # For CLIP visual encoder
            "expanded_mask": expanded_mask_tensor,  # For loss weighting
            "prompt": self.prompt
        }


def collate_fn(batch):
    """Collate function for dataloader"""
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "clip_images": torch.stack([item["clip_image"] for item in batch]),
        "expanded_masks": torch.stack([item["expanded_mask"] for item in batch]),
        "prompts": [item["prompt"] for item in batch]
    }


def add_lora_to_unet(unet, learning_rate):
    """Add LoRA adapter to U-Net"""
    for param in unet.parameters():
        param.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet.add_adapter(unet_lora_config)
    lora_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))

    return unet, lora_layers


class SimpleAnomalyTrainer:
    """
    Simplified Training class for Anomaly Synthesis

    Key differences from original Anomagic:
    - NO RFA (Region-Focused Attention)
    - NO IP-Adapter (no IPAttnProcessor)
    - NO I_mask dependency
    - Simple visual+text concat for condition
    - Standard U-Net cross attention
    """

    def __init__(self, args, accelerator):
        self.args = args
        self.accelerator = accelerator
        self.device = accelerator.device

        # Load models
        self.load_models()

        # Initialize optimizer
        self.setup_optimizer()

    def load_models(self):
        """Load all model components"""
        print(f"Loading base model from: {self.args.pretrained_model_name_or_path}")
        print(f"Loading VAE from: {self.args.vae_model_path}")

        # Load scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="scheduler"
        )

        # Load tokenizer and text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="text_encoder"
        )

        # Load VAE separately
        self.vae = AutoencoderKL.from_pretrained(self.args.vae_model_path)

        # Load U-Net (standard 4-channel SD, NO IP-Adapter processors)
        self.unet = UNet2DConditionModel.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="unet"
        )

        # Load CLIP image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.args.image_encoder_path
        )

        # Freeze encoder models
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)

        # Initialize Condition Encoder (replaces RFA + ImageProjModel + IP-Adapter)
        self.condition_encoder = ConditionEncoder(
            clip_visual_dim=self.image_encoder.config.projection_dim,  # 1024 for ViT-H/14
            clip_text_dim=self.text_encoder.config.hidden_size,  # 768 for SD 1.5
            cross_attention_dim=self.unet.config.cross_attention_dim,  # 768 for SD 1.5
            num_visual_tokens=self.args.num_visual_tokens,
        )

        # Add LoRA to U-Net
        self.unet, self.lora_layers = add_lora_to_unet(
            self.unet, self.args.learning_rate
        )

        # Set mixed precision dtype for frozen models
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move frozen models to device with reduced precision
        self.vae.to(self.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.device, dtype=self.weight_dtype)
        self.image_encoder.to(self.device, dtype=self.weight_dtype)

        # Keep trainable models in FP32
        self.unet.to(self.device)
        self.condition_encoder.to(self.device)

        print(f"Condition Encoder: {self.args.num_visual_tokens} visual tokens + 77 text tokens")
        print("NO RFA, NO IP-Adapter - using simple visual+text concat")

    def setup_optimizer(self):
        """Set up optimizer"""
        # Trainable parameters: ConditionEncoder + U-Net LoRA
        params_to_opt = itertools.chain(
            self.condition_encoder.parameters(),
            self.lora_layers,
        )

        self.optimizer = torch.optim.AdamW(
            params_to_opt,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

    def encode_text(self, prompts):
        """
        Encode text prompts using CLIP text encoder

        Args:
            prompts: List[str] - text prompts

        Returns:
            text_embeds: [batch, seq_len, hidden_size]
        """
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(self.device)

        with torch.no_grad():
            # Get full sequence output, not just pooled
            text_embeds = self.text_encoder(text_input_ids)[0]  # [batch, 77, 768]

        return text_embeds

    def encode_image(self, clip_images):
        """
        Encode images using CLIP visual encoder

        Args:
            clip_images: [batch, 3, 224, 224] - preprocessed images

        Returns:
            image_embeds: [batch, projection_dim] (e.g., [batch, 1024])
        """
        with torch.no_grad():
            outputs = self.image_encoder(clip_images.to(self.device, dtype=self.weight_dtype))
            # Use projected embedding (image_embeds), not last_hidden_state
            image_embeds = outputs.image_embeds  # [batch, 1024]

        return image_embeds

    def train_step(self, batch):
        """
        Single training step

        Pipeline:
        1. I_ref → CLIP visual encoder → image_embeds
        2. prompt → CLIP text encoder → text_embeds
        3. ConditionEncoder(image_embeds, text_embeds) → P_c
        4. U-Net(noisy_latents, timesteps, P_c) → noise_pred
        5. Masked MSE loss
        """
        with self.accelerator.accumulate(self.unet):
            # 1. Encode I_ref to latent space
            with torch.no_grad():
                latents = self.vae.encode(batch["images"].to(self.weight_dtype)).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor

            # 2. Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # 3. Sample random timesteps
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (bsz,), device=latents.device
            ).long()

            # 4. Add noise (z_t)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            # 5. Get image embeddings from CLIP visual encoder
            image_embeds = self.encode_image(batch["clip_images"])  # [batch, 1024]

            # 6. Get text embeddings from CLIP text encoder
            with torch.no_grad():
                text_embeds = self.encode_text(batch["prompts"])  # [batch, 77, 768]

            # 7. Generate condition P_c using ConditionEncoder
            # P_c = concat(P_a, P_b) where P_a is visual, P_b is text
            P_c = self.condition_encoder(
                image_embeds.float(),
                text_embeds.float()
            )  # [batch, num_visual_tokens + 77, 768]

            # 8. U-Net forward pass with P_c as encoder_hidden_states
            # Standard cross attention - no IP-Adapter, no decoupled attention
            noise_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=P_c,
            ).sample

            # 9. Compute masked MSE loss
            expanded_mask = batch["expanded_masks"].unsqueeze(1)  # [B, 1, 64, 64]

            loss = (F.mse_loss(noise_pred.float(), noise.float(), reduction="none") *
                    expanded_mask).mean([1, 2, 3]).mean()

            # 10. Backward pass
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.item()

    def save_checkpoint(self, global_step):
        """Save training checkpoint"""
        checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save condition encoder
        condition_encoder_path = os.path.join(checkpoint_dir, f"condition_encoder-{global_step}{self.args.ckpt_suffix}.bin")
        torch.save({
            "condition_encoder": self.condition_encoder.state_dict(),
        }, condition_encoder_path)

        # Save U-Net with LoRA
        unet_path = os.path.join(checkpoint_dir, f"unet-{global_step}{self.args.ckpt_suffix}.bin")
        torch.save({
            "unet": self.unet.state_dict(),
        }, unet_path)

        print(f"Checkpoint saved to {checkpoint_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Simplified Anomaly Synthesis")

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to AD_dataset_s_ROI folder",
    )
    parser.add_argument(
        "--defect_type",
        type=str,
        required=True,
        choices=["mounting_plate"],
        help="Defect type to train: 'mounting_plate'",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="SG161222/Realistic_Vision_V6.0_B1_noVAE",
        help="Path to pretrained SD model (noVAE version)",
    )
    parser.add_argument(
        "--vae_model_path",
        type=str,
        default="stabilityai/sd-vae-ft-mse",
        help="Path to VAE model",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        help="Path to CLIP ViT-H/14 image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size per device",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X steps",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num_visual_tokens",
        type=int,
        default=4,
        help="Number of visual tokens for condition (P_a)",
    )
    parser.add_argument(
        "--ckpt_suffix",
        type=str,
        default="",
        help="Suffix for checkpoint file names, e.g. _application",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=args.output_dir,
    )

    # Set random seed
    if args.seed is not None:
        set_seed(args.seed)

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Create dataset and dataloader
    train_dataset = CustomAnomalyDataset(
        data_root=args.data_root,
        defect_type=args.defect_type,
        size=args.resolution
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True
    )

    # Initialize trainer
    trainer = SimpleAnomalyTrainer(args, accelerator)

    # Prepare for distributed training
    (trainer.unet, trainer.condition_encoder,
     trainer.optimizer, train_dataloader) = accelerator.prepare(
        trainer.unet, trainer.condition_encoder,
        trainer.optimizer, train_dataloader
    )

    # Training loop
    global_step = 0
    for epoch in range(args.num_train_epochs):
        trainer.unet.train()
        trainer.condition_encoder.train()

        epoch_loss = 0
        for step, batch in enumerate(train_dataloader):
            start_time = time.perf_counter()

            loss = trainer.train_step(batch)
            epoch_loss += loss

            step_time = time.perf_counter() - start_time

            if accelerator.is_main_process:
                if global_step % 10 == 0:
                    print(f"Epoch {epoch}, Step {global_step}, Loss: {loss:.4f}, Time: {step_time:.2f}s")

            # Save checkpoint
            if accelerator.is_main_process and global_step % args.save_steps == 0 and global_step > 0:
                trainer.save_checkpoint(global_step)

            global_step += 1

        # Epoch summary
        if accelerator.is_main_process:
            avg_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

    # Save final checkpoint
    if accelerator.is_main_process:
        trainer.save_checkpoint(global_step)
        print("Training completed!")


if __name__ == "__main__":
    main()
