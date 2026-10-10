"""
ROI 영역의 생성 이미지를 원본 이미지 좌표로 복원하는 스크립트

- inpainting_mask.png : 검은 캔버스(원본 크기)의 ROI 위치에 배치 → 1024x1024 → output_dir/masks/
- generated.png       : 원본 컬러 이미지의 ROI 위치에 교체     → 1024x1024 → output_dir/images/
"""

import argparse

import cv2
import numpy as np
from pathlib import Path


# defect_type별 ROI 설정
ROI_CONFIG = {
    "mounting_plate": {"region": "region1", "roi": (564, 368, 680, 484)},  # 116x116
}

OUTPUT_SIZE = (1024, 1024)


def process_inpainting_mask(mask_path, original_image, roi_info):
    """
    inpainting_mask.png → 원본 크기 검은 캔버스의 ROI 위치에 배치 → 1024x1024

    Args:
        mask_path     : inpainting_mask.png 경로
        original_image: 원본 이미지 (크기 참조용, numpy array)
        roi_info      : (x_min, y_min, x_max, y_max)

    Returns:
        1024x1024 grayscale 마스크 이미지
    """
    x_min, y_min, x_max, y_max = roi_info
    roi_w = x_max - x_min
    roi_h = y_max - y_min

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"마스크를 불러올 수 없습니다: {mask_path}")

    mask_resized = cv2.resize(mask, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

    orig_h, orig_w = original_image.shape[:2]
    canvas = np.zeros((orig_h, orig_w), dtype=np.uint8)
    canvas[y_min:y_max, x_min:x_max] = mask_resized

    return cv2.resize(canvas, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)


def process_generated_image(generated_path, original_image, roi_info):
    """
    generated.png → 원본 컬러 이미지의 ROI 위치에 교체 → 1024x1024

    Args:
        generated_path: generated.png 경로
        original_image: 원본 컬러 이미지 (numpy array)
        roi_info      : (x_min, y_min, x_max, y_max)

    Returns:
        1024x1024 컬러 이미지
    """
    x_min, y_min, x_max, y_max = roi_info
    roi_w = x_max - x_min
    roi_h = y_max - y_min

    generated = cv2.imread(str(generated_path), cv2.IMREAD_COLOR)
    if generated is None:
        raise ValueError(f"생성 이미지를 불러올 수 없습니다: {generated_path}")

    generated_resized = cv2.resize(generated, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)

    result = original_image.copy()
    result[y_min:y_max, x_min:x_max] = generated_resized

    return cv2.resize(result, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ROI 생성 이미지를 원본 이미지 좌표로 복원"
    )
    parser.add_argument(
        "--defect_type",
        type=str,
        default="mounting_plate",
        choices=["mounting_plate"],
        help="Defect type: 'mounting_plate'",
    )
    parser.add_argument(
        "--generated_dir",
        type=str,
        required=True,
        help="생성 이미지 폴더 (cycle_*/step_*/image_*/generated.png)",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="원본 이미지 루트 폴더 (AD_dataset_s_ROI)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="결과 저장 폴더 (images/, masks/ 하위에 저장)",
    )
    parser.add_argument(
        "--origin_image",
        type=str,
        default=None,
        help="원본 이미지 경로 (기본: {data_root}/{defect_type}/test/Origin/000.png)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = ROI_CONFIG[args.defect_type]
    roi_info = config["roi"]
    region = config["region"]
    print(f"defect_type='{args.defect_type}' → {region}, ROI={roi_info}")

    # 원본 이미지 로드
    if args.origin_image is not None:
        origin_path = Path(args.origin_image)
    else:
        origin_path = Path(args.data_root) / args.defect_type / "test" / "Origin" / "000.png"

    original_image = cv2.imread(str(origin_path), cv2.IMREAD_COLOR)
    if original_image is None:
        raise ValueError(f"원본 이미지를 불러올 수 없습니다: {origin_path}")
    print(f"원본 이미지 로드: {origin_path} {original_image.shape}")

    # 출력 폴더 생성
    masks_dir = Path(args.output_dir) / "masks"
    images_dir = Path(args.output_dir) / "images"
    masks_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # 생성 이미지 폴더 순회
    generated_dir = Path(args.generated_dir)
    cycles = sorted(generated_dir.glob("cycle_*"))
    if not cycles:
        raise ValueError(f"cycle 폴더를 찾을 수 없습니다: {generated_dir}")

    mask_count = 0
    image_count = 0

    for cycle_dir in cycles:
        # inpainting_mask.png 처리
        mask_path = cycle_dir / "inpainting_mask.png"
        if mask_path.exists():
            result_mask = process_inpainting_mask(mask_path, original_image, roi_info)
            cv2.imwrite(str(masks_dir / f"{mask_count:03d}.png"), result_mask)
            mask_count += 1

        # generated.png 처리 (step_XX/image_XX/generated.png)
        for gen_path in sorted(cycle_dir.glob("step_*/image_*/generated.png")):
            result_img = process_generated_image(gen_path, original_image, roi_info)
            cv2.imwrite(str(images_dir / f"{image_count:03d}.png"), result_img)
            image_count += 1

    print("\n완료:")
    print(f"  masks  : {mask_count}개 → {masks_dir}")
    print(f"  images : {image_count}개 → {images_dir}")


if __name__ == "__main__":
    main()
