"""
ROI restoration utility for converting 512x512 ROI back to original 1024x1024 image
Based on Application guide.txt specifications:
- Defect 'mounting_plate': 512x512 -> 116x116 -> region1 in 1024x1024
"""

import cv2
import numpy as np
from PIL import Image
import torch


# ROI information for different defect types
# Based on Mask_generation/defect-ROI_to_original1.py
ROI_INFO = {
    'mounting_plate': {
        'region': (564, 368, 680, 484),  # (x_min, y_min, x_max, y_max) - 116x116
        'roi_size': (116, 116)
    }
}


def restore_roi_to_original(roi_image, roi_mask, original_image, defect_type):
    """
    Restore 512x512 ROI image and mask back to original 1024x1024 image.

    Args:
        roi_image: PIL Image or numpy array (512x512x3) - ROI anomaly image (I_output)
        roi_mask: PIL Image or numpy array (512x512 or 256x256) - ROI anomaly mask (final_refined_mask)
        original_image: PIL Image or numpy array (1024x1024x3) - original normal image (I_normal)
        defect_type: str - 'mounting_plate'

    Returns:
        result_image: numpy array (1024x1024x3) - original image with ROI pasted
        result_mask: numpy array (1024x1024) - full-size mask with ROI pasted
    """
    if defect_type not in ROI_INFO:
        raise ValueError(f"Invalid defect_type: {defect_type}. Must be 'mounting_plate'")

    # Get ROI information
    roi_info = ROI_INFO[defect_type]
    x_min, y_min, x_max, y_max = roi_info['region']
    target_width, target_height = roi_info['roi_size']

    # Convert inputs to numpy arrays
    if isinstance(roi_image, Image.Image):
        roi_image_np = np.array(roi_image)
    elif isinstance(roi_image, torch.Tensor):
        roi_image_np = roi_image.cpu().numpy()
        if roi_image_np.ndim == 3 and roi_image_np.shape[0] == 3:
            roi_image_np = roi_image_np.transpose(1, 2, 0)
        roi_image_np = (roi_image_np * 255).astype(np.uint8) if roi_image_np.max() <= 1.0 else roi_image_np.astype(np.uint8)
    else:
        roi_image_np = roi_image

    if isinstance(roi_mask, Image.Image):
        roi_mask_np = np.array(roi_mask)
    elif isinstance(roi_mask, torch.Tensor):
        roi_mask_np = roi_mask.cpu().numpy()
        if roi_mask_np.ndim == 3:
            roi_mask_np = roi_mask_np.squeeze()
        roi_mask_np = (roi_mask_np * 255).astype(np.uint8) if roi_mask_np.max() <= 1.0 else roi_mask_np.astype(np.uint8)
    else:
        roi_mask_np = roi_mask

    if isinstance(original_image, Image.Image):
        original_np = np.array(original_image)
    elif isinstance(original_image, torch.Tensor):
        original_np = original_image.cpu().numpy()
        if original_np.ndim == 3 and original_np.shape[0] == 3:
            original_np = original_np.transpose(1, 2, 0)
        original_np = (original_np * 255).astype(np.uint8) if original_np.max() <= 1.0 else original_np.astype(np.uint8)
    else:
        original_np = original_image

    # Ensure correct shapes and resize if needed
    if roi_image_np.shape[:2] != (512, 512):
        roi_image_np = cv2.resize(roi_image_np, (512, 512), interpolation=cv2.INTER_LINEAR)
    if roi_mask_np.shape[:2] != (512, 512):
        # MetaUAS returns 256x256 masks, resize to 512x512
        roi_mask_np = cv2.resize(roi_mask_np, (512, 512), interpolation=cv2.INTER_NEAREST)
    if original_np.shape[:2] != (1024, 1024):
        raise ValueError(f"Original image must be 1024x1024, got {original_np.shape}")

    # Resize ROI image and mask to target size
    roi_image_resized = cv2.resize(
        roi_image_np,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR
    )

    roi_mask_resized = cv2.resize(
        roi_mask_np,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR
    )

    # Create result arrays (copy of original)
    result_image = original_np.copy()
    result_mask = np.zeros((1024, 1024), dtype=np.uint8)

    # Paste ROI to original position
    result_image[y_min:y_max, x_min:x_max] = roi_image_resized
    result_mask[y_min:y_max, x_min:x_max] = roi_mask_resized

    return result_image, result_mask


def restore_roi_to_original_with_overlay(roi_image, roi_mask, original_image, defect_type, alpha=0.3):
    """
    Restore ROI with visualization overlay.

    Args:
        roi_image: PIL Image or numpy array (512x512x3) - ROI anomaly image
        roi_mask: PIL Image or numpy array (512x512) - ROI anomaly mask
        original_image: PIL Image or numpy array (1024x1024x3) - original normal image
        defect_type: str - 'mounting_plate'
        alpha: float - transparency for mask overlay (0-1)

    Returns:
        result_image: numpy array (1024x1024x3) - original image with ROI and mask overlay
        result_mask: numpy array (1024x1024) - full-size mask
    """
    result_image, result_mask = restore_roi_to_original(
        roi_image, roi_mask, original_image, defect_type
    )

    # Get ROI information for visualization
    roi_info = ROI_INFO[defect_type]
    x_min, y_min, x_max, y_max = roi_info['region']

    # Create visualization with mask overlay
    vis_image = result_image.copy()
    overlay = result_image.copy()

    # Apply red overlay where mask is active
    overlay[result_mask > 0] = [0, 0, 255]  # BGR: Red

    # Blend with original
    vis_image = cv2.addWeighted(result_image, 1 - alpha, overlay, alpha, 0)

    # Draw mask contours
    contours, _ = cv2.findContours(result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_image, contours, -1, (0, 0, 255), 2)

    # Draw ROI bounding box
    cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Add text label
    label = f"Defect Type: {defect_type}"
    cv2.putText(vis_image, label, (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return vis_image, result_mask


if __name__ == "__main__":
    # Test code
    import os

    # Create dummy data for testing
    roi_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    roi_mask = np.zeros((512, 512), dtype=np.uint8)
    roi_mask[100:200, 100:200] = 255  # Dummy anomaly region

    original_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

    # Test defect type 'mounting_plate'
    result_img, result_mask = restore_roi_to_original(
        roi_image, roi_mask, original_image, 'mounting_plate'
    )

    print("Test passed!")
    print(f"Result image shape: {result_img.shape}")
    print(f"Result mask shape: {result_mask.shape}")

    # Test with visualization
    vis_img, vis_mask = restore_roi_to_original_with_overlay(
        roi_image, roi_mask, original_image, 'mounting_plate'
    )

    print(f"Visualization image shape: {vis_img.shape}")
