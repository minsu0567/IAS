"""
Generation script for Simplified Anomaly Synthesis Pipeline (Multi-Seed Version)

Key features:
- NO RFA (Region-Focused Attention)
- NO IP-Adapter
- NO I_mask dependency for condition generation
- Simple visual+text concat approach (like anomalydiffusion-master)
- **Multi-seed generation: Generate multiple images per step with different seeds**

Uses StableDiffusionInpaintPipelineLegacy for inference with custom condition injection.
"""

import os
import argparse
import json
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
import warnings
import logging

# Suppress verbose outputs
warnings.filterwarnings('ignore')
logging.getLogger('diffusers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
logging.getLogger('peft').setLevel(logging.ERROR)
logging.getLogger('accelerate').setLevel(logging.ERROR)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['DIFFUSERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
Image.MAX_IMAGE_PIXELS = None

from diffusers import StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL, DPMSolverMultistepScheduler
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from peft import LoraConfig

from condition_encoder import ConditionEncoder


def get_generator(seed, device):
    """Get random generator for reproducibility"""
    if seed is not None:
        return torch.Generator(device=device).manual_seed(seed)
    return None


class SimpleAnomalyGenerator:
    """
    Simplified Anomaly Generator

    Uses simple visual+text concat for conditioning:
    - I_ref → CLIP Visual Encoder → Linear → P_a
    - prompt → CLIP Text Encoder → Linear → P_b
    - P_c = concat(P_a, P_b) → U-Net condition

    NO RFA, NO IP-Adapter, NO I_mask for condition generation.
    """

    def __init__(
        self,
        condition_encoder_ckpt: str,
        unet_ckpt: str,
        base_model_path: str,
        vae_model_path: str,
        image_encoder_path: str,
        device: str = "cuda",
        num_visual_tokens: int = 4,
        quiet: bool = False,
    ):
        self.device = torch.device(device)
        self.num_visual_tokens = num_visual_tokens
        self.quiet = quiet

        # Disable diffusers progress bar if quiet mode
        if quiet:
            from diffusers.utils import logging as diffusers_logging
            diffusers_logging.disable_progress_bar()

        # Load pipeline
        self._load_pipeline(base_model_path, vae_model_path)

        # Load CLIP image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_path
        ).to(self.device, dtype=torch.float16)
        self.image_encoder.eval()

        self.clip_image_processor = CLIPImageProcessor()

        # Initialize and load condition encoder
        self.condition_encoder = ConditionEncoder(
            clip_visual_dim=self.image_encoder.config.projection_dim,  # 1024
            clip_text_dim=self.pipe.text_encoder.config.hidden_size,  # 768
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,  # 768
            num_visual_tokens=num_visual_tokens,
        ).to(self.device, dtype=torch.float16)

        # Load checkpoints
        self._load_checkpoints(condition_encoder_ckpt, unet_ckpt)

        self.condition_encoder.eval()

    def _load_pipeline(self, base_model_path, vae_model_path):
        """Load Stable Diffusion pipeline"""
        # Configure scheduler
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        # Load VAE separately
        vae = AutoencoderKL.from_pretrained(vae_model_path).to(
            device=self.device, dtype=torch.float16
        )

        # Load Legacy Inpainting Pipeline
        self.pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None
        ).to(self.device)

        # Use DPMSolver for faster inference
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

    def _load_checkpoints(self, condition_encoder_ckpt, unet_ckpt):
        """Load trained checkpoints"""
        # Load condition encoder
        if not self.quiet:
            print(f"Loading condition encoder from: {condition_encoder_ckpt}")
        ce_state = torch.load(condition_encoder_ckpt, map_location="cpu")
        self.condition_encoder.load_state_dict(ce_state["condition_encoder"])

        # Add LoRA to U-Net and load weights
        if not self.quiet:
            print(f"Loading U-Net from: {unet_ckpt}")

        # Add LoRA config (same as training)
        unet_lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.pipe.unet.add_adapter(unet_lora_config)

        # Load U-Net weights
        unet_state = torch.load(unet_ckpt, map_location="cpu")
        self.pipe.unet.load_state_dict(unet_state["unet"], strict=False)

        if not self.quiet:
            print("Checkpoints loaded successfully")

    @torch.no_grad()
    def encode_image(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Encode image using CLIP visual encoder

        Args:
            pil_image: PIL Image (I_ref)

        Returns:
            image_embeds: [1, projection_dim] (e.g., [1, 1024])
        """
        clip_image = self.clip_image_processor(
            images=pil_image, return_tensors="pt"
        ).pixel_values.to(self.device, dtype=torch.float16)

        outputs = self.image_encoder(clip_image)
        return outputs.image_embeds  # [1, 1024]

    @torch.no_grad()
    def encode_text(self, prompt: str) -> torch.Tensor:
        """
        Encode text using CLIP text encoder

        Args:
            prompt: Text prompt

        Returns:
            text_embeds: [1, 77, hidden_size]
        """
        text_inputs = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(self.device)
        text_embeds = self.pipe.text_encoder(text_input_ids)[0]

        return text_embeds  # [1, 77, 768]

    @torch.no_grad()
    def generate(
        self,
        normal_image: Image.Image,
        inpainting_mask: Image.Image,
        ref_image: Image.Image,
        prompt: str,
        negative_prompt: str = "monochrome, lowres, bad anatomy, worst quality, low quality",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        strength: float = 0.8,
        seed: int = None,
    ) -> Image.Image:
        """
        Generate anomaly image

        Args:
            normal_image: PIL.Image - normal ROI image (I_normal_ROI)
            inpainting_mask: PIL.Image - mask for inpainting region (I_inpainting_mask)
            ref_image: PIL.Image - reference defect image (I_ref)
            prompt: str - text prompt describing the anomaly
            negative_prompt: str - negative prompt
            guidance_scale: float - CFG scale
            num_inference_steps: int - number of denoising steps
            strength: float - denoising strength (0-1)
            seed: int - random seed

        Returns:
            PIL.Image - generated anomaly image
        """
        # 1. Encode I_ref with CLIP visual encoder
        image_embeds = self.encode_image(ref_image)  # [1, 1024]

        # 2. Encode prompt with CLIP text encoder
        text_embeds = self.encode_text(prompt)  # [1, 77, 768]

        # 3. Generate condition P_c = concat(P_a, P_b)
        P_c = self.condition_encoder(
            image_embeds.half(),
            text_embeds.half()
        )  # [1, num_visual_tokens + 77, 768]

        # 4. Generate unconditioned embeddings for CFG
        uncond_text_embeds = self.encode_text(negative_prompt)
        uncond_image_embeds = torch.zeros_like(image_embeds)
        uncond_P_c = self.condition_encoder(
            uncond_image_embeds.half(),
            uncond_text_embeds.half()
        )

        # 5. Concatenate for CFG: [uncond, cond]
        prompt_embeds = P_c
        negative_prompt_embeds = uncond_P_c

        # 6. Generate with pipeline
        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image=normal_image,
            mask_image=inpainting_mask,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images

        return images[0]


class AnomalyTestRunner:
    """
    Test runner for anomaly generation
    Multi-seed version: generates multiple images per step
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize generator
        self.generator = SimpleAnomalyGenerator(
            condition_encoder_ckpt=args.condition_encoder_checkpoint,
            unet_ckpt=args.unet_checkpoint,
            base_model_path=args.pretrained_model_path,
            vae_model_path=args.vae_model_path,
            image_encoder_path=args.image_encoder_path,
            device=self.device,
            num_visual_tokens=args.num_visual_tokens,
            quiet=args.quiet,
        )

        # Set paths
        self.data_root = Path(args.data_root)
        self.defect_type = args.defect_type

        # Load reference images (I_ref) - NO I_mask needed
        self.load_reference_images()

        # Load normal image
        self.load_normal_image()

        # Load prompt
        prompt_file = self.data_root / self.defect_type / f"prompt_{self.defect_type}.txt"
        with open(prompt_file, 'r', encoding='utf-8') as f:
            self.prompt = f.read().strip()

        if not self.args.quiet:
            print(f"Ready: {len(self.ref_images)} references, device={self.device}")
            print("Using simplified pipeline: NO RFA, NO IP-Adapter, NO I_mask")
            print(f"Multi-seed mode: {args.images_per_step} images per step")

    def load_reference_images(self):
        """Load reference defect images (I_ref only, NO I_mask)"""
        train_dir = self.data_root / self.defect_type / "train"

        self.ref_images = []
        for img_file in sorted(train_dir.glob("*.png")):
            self.ref_images.append({
                'image': Image.open(img_file).convert("RGB"),
                'name': img_file.stem
            })

    def load_normal_image(self):
        """Load normal ROI and original images"""
        roi_dir = self.data_root / self.defect_type / "test" / "ROI"
        origin_dir = self.data_root / self.defect_type / "test" / "Origin"

        roi_files = list(roi_dir.glob("*.png"))
        origin_files = list(origin_dir.glob("*.png"))

        if len(roi_files) == 0 or len(origin_files) == 0:
            raise ValueError(f"No test images found in {roi_dir} or {origin_dir}")

        self.normal_roi = Image.open(roi_files[0]).convert("RGB")
        self.normal_origin = Image.open(origin_files[0]).convert("RGB")

    def generate_one_cycle(self, inpainting_mask, output_dir, cycle_idx):
        """
        Generate one cycle with multi-reference and multi-seed generation

        Each step uses a different I_ref and generates multiple images with different seeds

        Structure:
        - step 0 (ref_0): seed=S+0, S+1, ..., S+N → N images
        - step 1 (ref_1): seed=S+0, S+1, ..., S+N → N images
        - ...
        - step M (ref_M): seed=S+0, S+1, ..., S+N → N images

        Total: M steps × N images = M×N images per cycle
        """
        cycle_dir = output_dir / f"cycle_{cycle_idx:03d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)

        # Save inpainting mask
        inpainting_mask.save(cycle_dir / "inpainting_mask.png")

        total_images = 0

        if not self.args.quiet:
            print(f"\nProcessing Cycle {cycle_idx}...")

        for step_idx, ref_data in enumerate(self.ref_images):
            step_dir = cycle_dir / f"step_{step_idx:02d}"
            step_dir.mkdir(exist_ok=True)

            # Generate multiple images with different seeds for this step
            iterator = range(self.args.images_per_step)
            if not self.args.quiet:
                iterator = tqdm(iterator, desc=f"  Step {step_idx} ({ref_data['name']})", leave=False)

            for img_idx in iterator:
                # Calculate seed for this image (reset seed per step)
                if self.args.seed is not None:
                    current_seed = self.args.seed + img_idx
                else:
                    current_seed = None

                # Generate anomaly image
                generated_image = self.generator.generate(
                    normal_image=self.normal_roi,
                    inpainting_mask=inpainting_mask,
                    ref_image=ref_data['image'],  # I_ref only, NO I_mask
                    prompt=self.prompt,
                    negative_prompt=self.args.negative_prompt,
                    guidance_scale=self.args.guidance_scale,
                    num_inference_steps=self.args.num_inference_steps,
                    strength=self.args.strength,
                    seed=current_seed
                )

                # Save generated image
                img_dir = step_dir / f"image_{img_idx:02d}"
                img_dir.mkdir(exist_ok=True)
                generated_image.save(img_dir / "generated.png")

                total_images += 1

        if not self.args.quiet:
            print(f"\nCycle {cycle_idx} completed: {total_images} images generated")

        return {
            'cycle_idx': cycle_idx,
            'total_images': total_images,
            'cycle_dir': cycle_dir,
        }

    def run(self):
        """Run test generation"""
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load inpainting masks
        inpainting_mask_dir = self.data_root / self.defect_type / "inpainting_mask"
        inpainting_masks = sorted(list(inpainting_mask_dir.glob("*.png")))

        if len(inpainting_masks) == 0:
            raise ValueError(f"No inpainting masks found in {inpainting_mask_dir}")

        num_generations = min(self.args.num_generations, len(inpainting_masks))
        inpainting_masks = inpainting_masks[:num_generations]

        if not self.args.quiet:
            print(f"\nGenerating {num_generations} cycles...")
            print(f"Each cycle: {len(self.ref_images)} steps × {self.args.images_per_step} images = "
                  f"{len(self.ref_images) * self.args.images_per_step} images")
            print(f"Total: {num_generations * len(self.ref_images) * self.args.images_per_step} images\n")

        total_images = 0

        for cycle_idx, mask_path in enumerate(inpainting_masks):
            inpainting_mask = Image.open(mask_path).convert("L")

            result = self.generate_one_cycle(
                inpainting_mask=inpainting_mask,
                output_dir=output_dir,
                cycle_idx=cycle_idx
            )
            total_images += result['total_images']

        # Save summary
        summary = {
            'defect_type': self.defect_type,
            'num_cycles': num_generations,
            'num_steps_per_cycle': len(self.ref_images),
            'images_per_step': self.args.images_per_step,
            'images_per_cycle': len(self.ref_images) * self.args.images_per_step,
            'total_images': total_images,
            'pipeline': 'simplified (no RFA, no IP-Adapter)',
            'seed_strategy': 'reset per step',
        }

        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Generation Complete!")
        print(f"{'='*60}")
        print(f"Total images generated: {total_images}")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*60}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test script for Simplified Anomaly Synthesis (Multi-Seed)")

    # Data paths
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
        help="Defect type: 'mounting_plate'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_output_multi_seed",
        help="Output directory",
    )

    # Model paths
    parser.add_argument(
        "--pretrained_model_path",
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
        "--condition_encoder_checkpoint",
        type=str,
        required=True,
        help="Path to condition encoder checkpoint (condition_encoder-*.bin)",
    )
    parser.add_argument(
        "--unet_checkpoint",
        type=str,
        required=True,
        help="Path to U-Net checkpoint (unet-*.bin)",
    )

    # Generation parameters
    parser.add_argument(
        "--num_generations",
        type=int,
        default=10,
        help="Number of cycles (inpainting masks) to process",
    )
    parser.add_argument(
        "--images_per_step",
        type=int,
        default=10,
        help="Number of images to generate per step (with different seeds)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help="Denoising strength (0-1)",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="monochrome, lowres, bad anatomy, worst quality, low quality",
        help="Negative prompt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (will use seed, seed+1, seed+2, ... for each step)",
    )
    parser.add_argument(
        "--num_visual_tokens",
        type=int,
        default=4,
        help="Number of visual tokens (must match training)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimize output logs",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    runner = AnomalyTestRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
