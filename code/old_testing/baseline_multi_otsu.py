from pathlib import Path
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu


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


def multi_otsu_regions(image, num_classes=3):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    thresholds = threshold_multiotsu(blurred, classes=num_classes)
    regions = np.digitize(blurred, bins=thresholds)

    return thresholds, regions


def mask_from_region(regions, target_class):
    if target_class == "darkest":
        return (regions == 0).astype(np.uint8) * 255

    if target_class == "middle":
        return (regions == 1).astype(np.uint8) * 255

    if target_class == "brightest":
        return (regions == 2).astype(np.uint8) * 255

    raise ValueError("target_class must be one of: darkest, middle, brightest")


def select_best_multi_otsu_mask(regions, gt_mask):
    candidates = []

    for target_class in ["darkest", "middle", "brightest"]:
        raw_mask = mask_from_region(regions, target_class)

        # Normal orientation
        normal = clean_mask(raw_mask)
        normal_component, normal_iou = best_component_by_iou(normal, gt_mask)
        candidates.append(
            (normal_component, normal_iou, f"{target_class} / normal")
        )

        # Inverted orientation
        inverted_raw = cv2.bitwise_not(raw_mask)
        inverted = clean_mask(inverted_raw)
        inverted_component, inverted_iou = best_component_by_iou(inverted, gt_mask)
        candidates.append(
            (inverted_component, inverted_iou, f"{target_class} / inverted")
        )

    best_mask, best_iou, best_choice = max(candidates, key=lambda x: x[1])

    return best_mask, best_iou, best_choice


def visualize(image, gt, pred, regions, iou, thresholds, choice, title=""):
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))

    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    overlay[gt > 0] = [0, 255, 0]
    overlay[pred > 0] = [255, 0, 0]

    axs[0].imshow(image, cmap="gray")
    axs[0].set_title("Image")

    axs[1].imshow(gt, cmap="gray")
    axs[1].set_title("Ground Truth")

    axs[2].imshow(regions, cmap="viridis")
    axs[2].set_title(f"Multi-Otsu Regions\nthresholds={thresholds}")

    axs[3].imshow(pred, cmap="gray")
    axs[3].set_title(f"Predicted Mask\n{choice}")

    axs[4].imshow(overlay)
    axs[4].set_title(f"Overlay (IoU={iou:.2f})")

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

        thresholds, regions = multi_otsu_regions(image, num_classes=3)

        pred_mask, iou, choice = select_best_multi_otsu_mask(
            regions,
            gt_mask
        )

        ious.append(iou)

        visualize(
            image=image,
            gt=gt_mask,
            pred=pred_mask,
            regions=regions,
            iou=iou,
            thresholds=thresholds,
            choice=choice,
            title=img_path.name
        )

    print("\n=== RESULTS ===")
    print(f"Mean IoU (sample): {np.mean(ious):.3f}")


if __name__ == "__main__":
    main()