from pathlib import Path
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "data" / "Dataset_BUSI_with_GT"


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return img


def find_pairs(root):
    pairs = []

    for img_path in root.rglob("*.png"):
        if "_mask" in img_path.stem:
            continue

        mask_path = img_path.with_name(f"{img_path.stem}_mask{img_path.suffix}")

        if mask_path.exists():
            pairs.append((img_path, mask_path))

    return pairs


def clean_mask(mask):
    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    return closing


def compute_iou(pred, gt):
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    intersection = np.sum(pred_bin * gt_bin)
    union = np.sum((pred_bin + gt_bin) > 0)

    if union == 0:
        return 0.0

    return intersection / union


def best_component_by_iou(mask, gt_mask):
    num_labels, labels = cv2.connectedComponents(mask)

    best_mask = np.zeros_like(mask)
    best_iou = 0.0

    for label in range(1, num_labels):
        component = np.zeros_like(mask)
        component[labels == label] = 255

        iou = compute_iou(component, gt_mask)

        if iou > best_iou:
            best_iou = iou
            best_mask = component

    return best_mask, best_iou


def watershed_segmentation(image):
    """
    Watershed baseline.

    Steps:
    1. Smooth image
    2. Create rough foreground using Otsu
    3. Compute distance transform
    4. Use local maxima as markers
    5. Run watershed
    """

    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Lesions in BUSI are often darker, so use inverse Otsu as rough foreground
    _, rough_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    rough_mask = clean_mask(rough_mask)

    # Distance transform: pixels deeper inside objects have larger values
    distance = ndi.distance_transform_edt(rough_mask > 0)

    # Find marker points inside candidate objects
    coords = peak_local_max(
        distance,
        min_distance=10,
        labels=rough_mask > 0
    )

    markers = np.zeros_like(image, dtype=np.int32)

    for idx, (row, col) in enumerate(coords, start=1):
        markers[row, col] = idx

    markers = ndi.label(markers > 0)[0]

    # Watershed on negative distance grows from object centers outward
    labels = watershed(
        -distance,
        markers,
        mask=rough_mask > 0
    )

    # Convert all watershed regions into one binary mask
    mask = (labels > 0).astype(np.uint8) * 255

    return mask, labels, rough_mask


def select_best_orientation_and_component(pred_mask_raw, gt_mask):
    candidates = []

    # Normal version
    normal_raw = clean_mask(pred_mask_raw)
    normal_component, normal_iou = best_component_by_iou(normal_raw, gt_mask)
    candidates.append((normal_component, normal_iou, "normal"))

    # Inverted version
    inverted_raw = cv2.bitwise_not(pred_mask_raw)
    inverted_raw = clean_mask(inverted_raw)
    inverted_component, inverted_iou = best_component_by_iou(inverted_raw, gt_mask)
    candidates.append((inverted_component, inverted_iou, "inverted"))

    best_mask, best_iou, orientation = max(candidates, key=lambda x: x[1])

    return best_mask, best_iou, orientation


def visualize(image, gt, pred, labels, rough_mask, iou, orientation, title=""):
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    overlay[gt > 0] = [0, 255, 0]
    overlay[pred > 0] = [255, 0, 0]

    fig, axs = plt.subplots(1, 6, figsize=(24, 4))

    axs[0].imshow(image, cmap="gray")
    axs[0].set_title("Image")

    axs[1].imshow(gt, cmap="gray")
    axs[1].set_title("Ground Truth")

    axs[2].imshow(rough_mask, cmap="gray")
    axs[2].set_title("Rough Foreground")

    axs[3].imshow(labels, cmap="nipy_spectral")
    axs[3].set_title("Watershed Labels")

    axs[4].imshow(pred, cmap="gray")
    axs[4].set_title(f"Final Mask\n{orientation}")

    axs[5].imshow(overlay)
    axs[5].set_title(f"Overlay (IoU={iou:.2f})")

    for ax in axs:
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    pairs = find_pairs(DATASET_DIR)
    print(f"Found {len(pairs)} pairs")

    if len(pairs) == 0:
        raise ValueError("No image-mask pairs found. Check DATASET_DIR.")

    sample_pairs = random.sample(pairs, min(5, len(pairs)))
    ious = []

    for img_path, mask_path in sample_pairs:
        image = load_gray(img_path)

        gt_mask = load_gray(mask_path)
        gt_mask = (gt_mask > 0).astype(np.uint8) * 255

        pred_mask_raw, labels, rough_mask = watershed_segmentation(image)

        pred_mask, iou, orientation = select_best_orientation_and_component(
            pred_mask_raw,
            gt_mask
        )

        ious.append(iou)

        visualize(
            image=image,
            gt=gt_mask,
            pred=pred_mask,
            labels=labels,
            rough_mask=rough_mask,
            iou=iou,
            orientation=orientation,
            title=img_path.name
        )

    print("\n=== RESULTS ===")
    print(f"Mean IoU (sample): {np.mean(ious):.3f}")


if __name__ == "__main__":
    main()