# IAS

Generating realistic defect images to improve anomaly localization in industrial inspection.

A Stable Diffusion inpainting model (Realistic Vision + LoRA) is conditioned on a reference defect image through a
CLIP-based condition encoder and paints synthetic defects into the ROI of normal images. The synthesized
image / mask pairs are restored to the original image coordinates and used to train **DRAEM**, whose
pixel-level localization is compared with **PatchCore** and **EfficientAD**.

```
normal image + inpainting mask + reference defect
      │
[Generation]       Realistic Vision SD inpainting + LoRA + condition encoder (CLIP ViT-H/14 visual tokens)
      │
[Pre-processing]   ROI result → original image coordinates (1024×1024 image + mask)
      │
[Downstream]       DRAEM training on synthesized defects → image / pixel AUROC, AP, F1-max
      │
[Analysis]         pixel-level anomaly score histograms: DRAEM (ours) / PatchCore / EfficientAD
                   heatmaps normalized by each method's pixel F1-max threshold (over-segmentation vs. GT)
```

## Model weights

| Model | Weights |
|---|---|
| Realistic Vision SD — condition encoder | [condition_encoder-600_application.bin](https://drive.google.com/file/d/1fXAMLPEJ13aOjTSjArMY3fhJ13xwQhqM/view?usp=drive_link) |
| Realistic Vision SD — U-Net (LoRA) | [unet-600_application.bin](https://drive.google.com/file/d/189lEmB33dvHX1zAQNe1UTSf8zTnxhe9o/view?usp=drive_link) |
| DRAEM — reconstructive | [draem_rec_399.pckl](https://drive.google.com/file/d/1Uf5dvx_lYvEXQmmtX-4YNAvqD26bqJ7b/view?usp=drive_link) |
| DRAEM — discriminative | [draem_seg_399.pckl](https://drive.google.com/file/d/1PneIK3TsUYWl7jmnoqVgVSNqyuZq4oey/view?usp=drive_link) |

Base models are pulled from Hugging Face at run time:
[SG161222/Realistic_Vision_V6.0_B1_noVAE](https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE),
[stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse),
[laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K).

## Layout

```
Project-IAS/
├── generation_src/
│   ├── condition_encoder.py                CLIP image features → visual tokens, concatenated with text tokens
│   ├── train.py                            SD inpainting + LoRA + condition encoder training
│   ├── generation.py                       defect synthesis (single seed)
│   ├── generation_multi_seed.py            defect synthesis over multiple seeds
│   └── eval_kid_bbox_only.py               KID on bounding-box crops (real vs. generated defects)
├── pre-processing/
│   ├── roi_utils.py                        ROI coordinates per defect type
│   ├── restore_generated_to_original.py    generated ROI → original image / mask (1024×1024)
│   └── restore_roi_batch.py                batch restore with overlay
├── downstream_task_src/
│   ├── models/                             DRAEM networks and losses (Focal, SSIM, Dice)
│   ├── train_draem.py                      DRAEM training on good + synthesized defect images
│   └── eval_draem.py                       image / pixel AUROC, AP, F1-max, anomaly maps
├── score_histogram_Dream-ours.py           pixel-level score histogram — DRAEM (ours)
├── score_histogram_PatchCore.py            pixel-level score histogram — PatchCore
├── score_histogram_EfficientAD.py          pixel-level score histogram — EfficientAD
└── notebooks/                              Colab drivers: install, train, copy to Drive, run
```

## Google Drive

The notebooks run on Colab and read everything from Drive. Copy this repository to
`MyDrive/Project-IAS/`; these siblings must exist:

```
MyDrive/
├── Project-IAS/                            this repository
│   └── model/                              trained weights (written by the notebooks)
│       ├── stable_diffusion/               condition_encoder-600_application.bin, unet-600_application.bin
│       └── draem/                          draem_rec_399.pckl, draem_seg_399.pckl
├── AD_dataset_s_ROI/<defect_type>/         generation data
│   ├── train/                              reference defect images
│   ├── ground_truth/                       <name>_mask.png
│   ├── test/ROI/ · test/Origin/            ROI crops and original images to paint into
│   ├── inpainting_mask/
│   └── prompt_<defect_type>.txt
└── AD_dataset_application3/<defect_type>/  DRAEM data
    ├── train/good/ · train/defect/         normal + synthesized defect images
    ├── inpainting_mask/                    masks of the synthesized defects
    ├── test/good/ · test/defect/
    └── ground_truth/
```

The datasets are not included in this repository. The default defect type is `mounting_plate`.

## Notebooks

| # | Notebook | Purpose |
|---|---|---|
| 1 | `IAS_application.ipynb` | train SD → copy checkpoints to Drive → multi-seed generation → ROI restore → visual check |
| 2 | `Downstream_task_application.ipynb` | train DRAEM → copy checkpoints to Drive → evaluate → print metrics |
| 3 | `heatmap_three_methods.ipynb` | EfficientAD / PatchCore / DRAEM heatmaps on one scale (1.0 = each method's pixel F1-max threshold), Pred/GT area ratio |

`heatmap_three_methods.ipynb` reads raw float anomaly maps (`{subdir}/{name}.tiff`) of the three methods,
aligns them to the PatchCore `Resize → CenterCrop` frame and computes each method's pixel F1-max threshold on
that common frame. The F1-max threshold is chosen on the test set (oracle), so it is for comparison only.

Training writes checkpoints to the Colab local disk (`/content/sd_train_output`, `/content/draem_train_output`);
the next cell copies them to `MyDrive/Project-IAS/model/` with `!cp -r`, and generation / evaluation read them
from there.

```python
# IAS_application.ipynb
!cp -r {TRAIN_OUTPUT_DIR}/checkpoint-{CKPT_STEP}/*{CKPT_SUFFIX}.bin {SD_CKPT_DIR}/
```

`train.py --ckpt_suffix _application` names the files `condition_encoder-600_application.bin` /
`unet-600_application.bin` (default: no suffix).

## Environments

Every notebook installs its own packages in its first cell (CUDA 11.8 PyTorch, diffusers, transformers,
accelerate, peft, opencv-python). A GPU runtime is required.

The score-histogram scripts run inside the baseline repositories: `score_histogram_PatchCore.py` goes in
`bin/` of patchcore-inspection (it imports `../src/patchcore`), and `score_histogram_EfficientAD.py` imports
EfficientAD's `common.py` from `/content/EfficientAD` on Colab or from its own directory locally.

## Reference

- **[DRAEM](https://github.com/VitjanZ/DRAEM)** — `downstream_task_src/models/` follows its reconstructive /
  discriminative networks and losses
- **[PatchCore](https://github.com/amazon-science/patchcore-inspection)** and
  **[EfficientAD](https://github.com/nelson1425/EfficientAD)** — baselines for the score histograms
