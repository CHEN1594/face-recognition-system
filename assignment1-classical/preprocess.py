import argparse
import os
from pathlib import Path

import cv2
import numpy as np


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def is_low_quality(image: np.ndarray, min_std: float, min_laplacian_var: float) -> bool:
    """
    Basic quality gate:
    - Very low contrast images are likely unusable.
    - Very blurry images are likely unusable.从来
    """
    std_val = float(np.std(image))
    lap_var = float(cv2.Laplacian(image, cv2.CV_64F).var())
    return std_val < min_std or lap_var < min_laplacian_var


def preprocess_face(gray_image: np.ndarray, use_clahe: bool, normalize_mode: str) -> np.ndarray:
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(gray_image)
    else:
        processed = cv2.equalizeHist(gray_image)

    if normalize_mode == "minmax":
        normalized = cv2.normalize(processed.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
    else:
        x = processed.astype(np.float32)
        mean = float(np.mean(x))
        std = float(np.std(x))
        normalized = (x - mean) / (std + 1e-6)
        # Map approximately to [0,1] for safe image saving and display.
        normalized = np.clip((normalized + 3.0) / 6.0, 0.0, 1.0)

    # Save back to uint8 image while keeping normalized data behavior in pipeline.
    return (normalized * 255.0).astype(np.uint8)


def preprocess_gallery(
    input_dir: Path,
    output_dir: Path,
    use_clahe: bool,
    normalize_mode: str,
    min_std: float,
    min_laplacian_var: float,
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    people_dirs = [p for p in sorted(input_dir.iterdir()) if p.is_dir()]
    if not people_dirs:
        print(f"No person folders found in: {input_dir}")
        return

    total_in = 0
    total_saved = 0
    total_filtered = 0

    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Method: {'CLAHE' if use_clahe else 'Histogram Equalization'} + {normalize_mode}")
    print("-" * 60)

    for person_dir in people_dirs:
        dst_person = output_dir / person_dir.name
        dst_person.mkdir(parents=True, exist_ok=True)

        person_in = 0
        person_saved = 0
        person_filtered = 0

        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            person_in += 1
            total_in += 1

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                person_filtered += 1
                total_filtered += 1
                continue

            # Keep the assignment target resolution fixed.
            if img.shape != (90, 90):
                img = cv2.resize(img, (90, 90), interpolation=cv2.INTER_AREA)

            if is_low_quality(img, min_std=min_std, min_laplacian_var=min_laplacian_var):
                person_filtered += 1
                total_filtered += 1
                continue

            out_img = preprocess_face(img, use_clahe=use_clahe, normalize_mode=normalize_mode)
            out_path = dst_person / img_path.name
            cv2.imwrite(str(out_path), out_img)

            person_saved += 1
            total_saved += 1

        print(
            f"{person_dir.name:<15} in={person_in:<3} saved={person_saved:<3} filtered={person_filtered:<3}"
        )

    print("-" * 60)
    print(f"Total in={total_in}, saved={total_saved}, filtered={total_filtered}")
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess gallery face images for classical recognition.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("../gallery"),
        help="Input gallery directory (default: ../gallery)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../gallery_preprocessed"),
        help="Output directory (default: ../gallery_preprocessed)",
    )
    parser.add_argument(
        "--method",
        choices=["clahe", "hist-eq"],
        default="clahe",
        help="Illumination normalization method.",
    )
    parser.add_argument(
        "--normalize",
        choices=["minmax", "zscore"],
        default="minmax",
        help="Pixel normalization mode.",
    )
    parser.add_argument(
        "--min-std",
        type=float,
        default=8.0,
        help="Minimum intensity std to keep an image.",
    )
    parser.add_argument(
        "--min-laplacian-var",
        type=float,
        default=20.0,
        help="Minimum Laplacian variance to keep an image (blur filter).",
    )
    args = parser.parse_args()

    preprocess_gallery(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        use_clahe=(args.method == "clahe"),
        normalize_mode=args.normalize,
        min_std=args.min_std,
        min_laplacian_var=args.min_laplacian_var,
    )


if __name__ == "__main__":
    main()
