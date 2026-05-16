import os
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_grayscale_image(path: Path) -> np.ndarray:
    """
    Loads an image as grayscale using OpenCV.
    Returns a 2D numpy array.
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def find_busi_pairs(dataset_root: Path):
    """
    Finds BUSI image-mask pairs.

    Expected BUSI naming pattern:
    - Image: benign (1).png
    - Mask : benign (1)_mask.png

    Works recursively inside dataset_root.
    """
    image_mask_pairs = []

    all_pngs = list(dataset_root.rglob("*.png"))

    # Keep only base images, not masks
    image_files = [p for p in all_pngs if "_mask" not in p.stem]

    for image_path in image_files:
        mask_path = image_path.with_name(f"{image_path.stem}_mask{image_path.suffix}")

        if mask_path.exists():
            image_mask_pairs.append((image_path, mask_path))
        else:
            print(f"[WARNING] No mask found for image: {image_path}")

    return image_mask_pairs


def summarize_dataset(pairs):
    """
    Prints basic information about the dataset pairs.
    """
    print("\n=== DATASET SUMMARY ===")
    print(f"Total image-mask pairs found: {len(pairs)}")

    category_counts = {}

    for image_path, _ in pairs:
        parent_folder = image_path.parent.name
        category_counts[parent_folder] = category_counts.get(parent_folder, 0) + 1

    print("\nSamples per category:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")


def inspect_sample(image_path: Path, mask_path: Path):
    """
    Loads one image-mask pair and prints basic properties.
    """
    image = load_grayscale_image(image_path)
    mask = load_grayscale_image(mask_path)

    print("\n=== SAMPLE INSPECTION ===")
    print(f"Image path: {image_path}")
    print(f"Mask path : {mask_path}")
    print(f"Image shape: {image.shape}")
    print(f"Mask shape : {mask.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Mask dtype : {mask.dtype}")
    print(f"Image intensity range: [{image.min()}, {image.max()}]")
    print(f"Mask unique values    : {np.unique(mask)}")

    return image, mask


def create_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Creates an RGB overlay where the mask is shown in red
    on top of the grayscale image.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Normalize mask to binary
    binary_mask = (mask > 0).astype(np.uint8)

    overlay = image_rgb.copy()
    overlay[binary_mask == 1] = [255, 0, 0]  # red mask

    blended = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)
    return blended


def visualize_samples(pairs, num_samples=5):
    """
    Randomly visualizes a few samples from the dataset.
    """
    if len(pairs) == 0:
        print("[ERROR] No pairs found to visualize.")
        return

    num_samples = min(num_samples, len(pairs))
    selected_pairs = random.sample(pairs, num_samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    # If only one sample, axes is 1D, so make it 2D for consistency
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, (image_path, mask_path) in enumerate(selected_pairs):
        image = load_grayscale_image(image_path)
        mask = load_grayscale_image(mask_path)
        overlay = create_overlay(image, mask)

        axes[row_idx, 0].imshow(image, cmap="gray")
        axes[row_idx, 0].set_title(f"Image\n{image_path.name}")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(mask, cmap="gray")
        axes[row_idx, 1].set_title(f"Mask\n{mask_path.name}")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_title("Overlay")
        axes[row_idx, 2].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    # Change this path to wherever you keep the BUSI dataset
    BASE_DIR = Path(__file__).resolve().parent.parent
    dataset_root = BASE_DIR / "data" / "Dataset_BUSI_with_GT"

    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_root}\n"
            "Update the dataset_root path in the script."
        )

    pairs = find_busi_pairs(dataset_root)

    if len(pairs) == 0:
        raise ValueError(
            "No image-mask pairs found. Check dataset structure and file names."
        )

    summarize_dataset(pairs)

    # Inspect one sample in detail
    sample_image_path, sample_mask_path = pairs[0]
    inspect_sample(sample_image_path, sample_mask_path)

    # Visualize random samples
    visualize_samples(pairs, num_samples=5)


if __name__ == "__main__":
    main()