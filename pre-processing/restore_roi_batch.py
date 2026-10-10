"""
Batch restore ROI results to original image size

This script processes multiple step directories and restores them to original size.

Usage:
    # Restore specific steps from a cycle
    python restore_roi_batch.py --cycle_dir test_output/cycle_000 --defect_type mounting_plate --normal_origin data/AD_dataset_s_ROI/mounting_plate/test/Origin/000.png

    # Restore specific step indices
    python restore_roi_batch.py --cycle_dir test_output/cycle_000 --defect_type mounting_plate --normal_origin data/AD_dataset_s_ROI/mounting_plate/test/Origin/000.png --step_indices 0 3 5

    # Restore best N steps based on metrics
    python restore_roi_batch.py --cycle_dir test_output/cycle_000 --defect_type mounting_plate --normal_origin data/AD_dataset_s_ROI/mounting_plate/test/Origin/000.png --top_n 3 --metric mean_confidence
"""

import argparse
from pathlib import Path
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

from roi_utils import restore_roi_to_original, restore_roi_to_original_with_overlay


def load_step_metrics(step_dir):
    """Load metrics from a step directory"""
    metrics_path = step_dir / "metrics.json"
    if not metrics_path.exists():
        return None

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    return metrics


def restore_single_step(step_dir, defect_type, normal_origin, output_dir, overlay_alpha=0.3):
    """Restore a single step to original image size"""
    # Load ROI image and mask
    roi_image_path = step_dir / "generated.png"
    roi_mask_path = step_dir / "refined_mask.png"

    if not roi_image_path.exists() or not roi_mask_path.exists():
        print(f"Warning: Missing files in {step_dir}")
        return None

    roi_image = Image.open(roi_image_path).convert("RGB")
    roi_mask_pil = Image.open(roi_mask_path).convert("L")
    normal_origin_pil = Image.open(normal_origin).convert("RGB")

    # Convert mask to binary numpy array
    roi_mask = (np.array(roi_mask_pil) > 128).astype(np.uint8)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Restore ROI to original
    result_image, result_mask = restore_roi_to_original(
        roi_image=roi_image,
        roi_mask=roi_mask,
        original_image=normal_origin_pil,
        defect_type=defect_type
    )

    # Save results
    Image.fromarray(result_image).save(output_dir / "final_image.png")
    Image.fromarray(result_mask).save(output_dir / "final_mask.png")

    # Create visualization with overlay
    vis_image, _ = restore_roi_to_original_with_overlay(
        roi_image=roi_image,
        roi_mask=roi_mask,
        original_image=normal_origin_pil,
        defect_type=defect_type,
        alpha=overlay_alpha
    )

    Image.fromarray(vis_image).save(output_dir / "final_visualization.png")

    # Load metrics
    metrics = load_step_metrics(step_dir)

    # Save restoration info
    info = {
        'step_dir': str(step_dir),
        'defect_type': defect_type,
        'metrics': metrics
    }

    with open(output_dir / "restoration_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Batch restore ROI results to original image size")

    parser.add_argument(
        "--cycle_dir",
        type=str,
        required=True,
        help="Path to cycle directory (e.g., test_output/cycle_000)",
    )
    parser.add_argument(
        "--defect_type",
        type=str,
        required=True,
        choices=["mounting_plate"],
        help="Defect type: 'mounting_plate'",
    )
    parser.add_argument(
        "--normal_origin",
        type=str,
        required=True,
        help="Path to normal origin image (I_normal)",
    )
    parser.add_argument(
        "--step_indices",
        type=int,
        nargs='+',
        default=None,
        help="Specific step indices to restore (e.g., 0 3 5). If not specified, all steps will be restored.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=None,
        help="Restore top N steps based on a metric (requires --metric)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["fill_rate", "mean_confidence"],
        default="mean_confidence",
        help="Metric to use for selecting top N steps",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: cycle_dir/restored)",
    )
    parser.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.3,
        help="Alpha value for overlay visualization (0-1)",
    )

    args = parser.parse_args()

    # Convert to Path objects
    cycle_dir = Path(args.cycle_dir)
    normal_origin = Path(args.normal_origin)

    # Validate paths
    if not cycle_dir.exists():
        raise ValueError(f"Cycle directory not found: {cycle_dir}")
    if not normal_origin.exists():
        raise ValueError(f"Normal origin image not found: {normal_origin}")

    # Get all step directories
    step_dirs = sorted(cycle_dir.glob("step_*"))

    if len(step_dirs) == 0:
        raise ValueError(f"No step directories found in {cycle_dir}")

    print(f"Found {len(step_dirs)} step directories")

    # Select steps to restore
    if args.step_indices is not None:
        # Restore specific step indices
        selected_steps = []
        for idx in args.step_indices:
            step_dir = cycle_dir / f"step_{idx:02d}"
            if step_dir.exists():
                selected_steps.append(step_dir)
            else:
                print(f"Warning: Step directory not found: {step_dir}")

        print(f"Restoring {len(selected_steps)} specified steps")

    elif args.top_n is not None:
        # Load all metrics and select top N
        step_metrics = []
        for step_dir in step_dirs:
            metrics = load_step_metrics(step_dir)
            if metrics is not None:
                step_metrics.append({
                    'step_dir': step_dir,
                    'metrics': metrics
                })

        # Sort by metric
        step_metrics.sort(key=lambda x: x['metrics'][args.metric], reverse=True)

        # Select top N
        selected_steps = [x['step_dir'] for x in step_metrics[:args.top_n]]

        print(f"Restoring top {args.top_n} steps based on {args.metric}")
        for i, item in enumerate(step_metrics[:args.top_n]):
            print(f"  {i+1}. {item['step_dir'].name}: {args.metric}={item['metrics'][args.metric]:.3f}")

    else:
        # Restore all steps
        selected_steps = step_dirs
        print(f"Restoring all {len(selected_steps)} steps")

    # Set output directory
    if args.output_dir is None:
        base_output_dir = cycle_dir / "restored"
    else:
        base_output_dir = Path(args.output_dir)

    # Restore each selected step
    restored_count = 0
    for step_dir in tqdm(selected_steps, desc="Restoring steps"):
        step_name = step_dir.name
        output_dir = base_output_dir / step_name

        metrics = restore_single_step(
            step_dir=step_dir,
            defect_type=args.defect_type,
            normal_origin=normal_origin,
            output_dir=output_dir,
            overlay_alpha=args.overlay_alpha
        )

        if metrics is not None:
            restored_count += 1

    # Save batch summary
    summary = {
        'cycle_dir': str(cycle_dir),
        'defect_type': args.defect_type,
        'normal_origin': str(normal_origin),
        'total_steps': len(step_dirs),
        'restored_steps': restored_count,
        'selection_method': 'all' if args.step_indices is None and args.top_n is None else ('indices' if args.step_indices is not None else f'top_{args.top_n}_{args.metric}'),
        'output_dir': str(base_output_dir)
    }

    summary_path = base_output_dir / "batch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("Batch restoration completed!")
    print(f"{'='*60}")
    print(f"Total steps processed: {restored_count}/{len(selected_steps)}")
    print(f"Results saved to: {base_output_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
