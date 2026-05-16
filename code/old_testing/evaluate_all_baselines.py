from pathlib import Path
import random
import csv

import cv2
import numpy as np
from skimage.filters import threshold_multiotsu
from skimage.segmentation import chan_vese, watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "data" / "Dataset_BUSI_with_GT"
OUTPUT_DIR = BASE_DIR / "evaluation"
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_IMAGES = 100
RANDOM_SEED = 42


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


def select_best_orientation_and_component(raw_mask, gt_mask):
    candidates = []

    normal = clean_mask(raw_mask)
    normal_component, normal_iou = best_component_by_iou(normal, gt_mask)
    candidates.append((normal_component, normal_iou, "normal"))

    inverted = cv2.bitwise_not(raw_mask)
    inverted = clean_mask(inverted)
    inverted_component, inverted_iou = best_component_by_iou(inverted, gt_mask)
    candidates.append((inverted_component, inverted_iou, "inverted"))

    return max(candidates, key=lambda x: x[1])


def otsu_raw(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return mask


def adaptive_raw(image, block_size=21, c_value=7):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c_value
    )

    return mask


def multi_otsu_best(image, gt_mask, num_classes=3):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    thresholds = threshold_multiotsu(blurred, classes=num_classes)
    regions = np.digitize(blurred, bins=thresholds)

    candidates = []

    for class_idx in range(num_classes):
        raw_mask = (regions == class_idx).astype(np.uint8) * 255

        normal = clean_mask(raw_mask)
        normal_component, normal_iou = best_component_by_iou(normal, gt_mask)
        candidates.append((normal_component, normal_iou, f"class_{class_idx}_normal"))

        inverted = cv2.bitwise_not(raw_mask)
        inverted = clean_mask(inverted)
        inverted_component, inverted_iou = best_component_by_iou(inverted, gt_mask)
        candidates.append((inverted_component, inverted_iou, f"class_{class_idx}_inverted"))

    return max(candidates, key=lambda x: x[1])


def chanvese_raw(image):
    image_float = image.astype(np.float32) / 255.0

    cv_mask = chan_vese(
        image_float,
        mu=0.25,
        lambda1=1,
        lambda2=1,
        tol=1e-3,
        max_num_iter=200,
        dt=0.5,
        init_level_set="checkerboard"
    )

    return cv_mask.astype(np.uint8) * 255


def watershed_raw(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    _, rough_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    rough_mask = clean_mask(rough_mask)

    distance = ndi.distance_transform_edt(rough_mask > 0)

    coords = peak_local_max(
        distance,
        min_distance=10,
        labels=rough_mask > 0
    )

    markers = np.zeros_like(image, dtype=np.int32)

    for idx, (row, col) in enumerate(coords, start=1):
        markers[row, col] = idx

    markers = ndi.label(markers > 0)[0]

    labels = watershed(
        -distance,
        markers,
        mask=rough_mask > 0
    )

    mask = (labels > 0).astype(np.uint8) * 255

    return mask


def evaluate_method(method_name, image, gt_mask):
    if method_name == "otsu":
        raw = otsu_raw(image)
        _, iou, choice = select_best_orientation_and_component(raw, gt_mask)
        return iou, choice

    if method_name == "multi_otsu":
        _, iou, choice = multi_otsu_best(image, gt_mask)
        return iou, choice

    if method_name == "adaptive":
        raw = adaptive_raw(image)
        _, iou, choice = select_best_orientation_and_component(raw, gt_mask)
        return iou, choice

    if method_name == "chan_vese":
        raw = chanvese_raw(image)
        _, iou, choice = select_best_orientation_and_component(raw, gt_mask)
        return iou, choice

    if method_name == "watershed":
        raw = watershed_raw(image)
        _, iou, choice = select_best_orientation_and_component(raw, gt_mask)
        return iou, choice

    raise ValueError(f"Unknown method: {method_name}")


def summarize(values):
    values = np.array(values, dtype=np.float32)

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main():
    random.seed(RANDOM_SEED)

    pairs = find_pairs(DATASET_DIR)
    print(f"Total image-mask pairs found: {len(pairs)}")

    positive_pairs = []

    for img_path, mask_path in pairs:
        gt_mask = load_gray(mask_path)
        gt_mask = (gt_mask > 0).astype(np.uint8) * 255

        # Exclude expert masks that are entirely black
        if np.sum(gt_mask > 0) == 0:
            continue

        positive_pairs.append((img_path, mask_path))

    print(f"Pairs with non-empty expert masks: {len(positive_pairs)}")

    sample_pairs = random.sample(
        positive_pairs,
        min(NUM_IMAGES, len(positive_pairs))
    )

    methods = [
        "otsu",
        "multi_otsu",
        "adaptive",
        "chan_vese",
        "watershed",
    ]

    results = []
    method_ious = {method: [] for method in methods}

    for idx, (img_path, mask_path) in enumerate(sample_pairs, start=1):
        print(f"[{idx}/{len(sample_pairs)}] Processing {img_path.name}")

        image = load_gray(img_path)

        gt_mask = load_gray(mask_path)
        gt_mask = (gt_mask > 0).astype(np.uint8) * 255

        row = {
            "image": img_path.name,
            "mask": mask_path.name,
        }

        for method in methods:
            try:
                iou, choice = evaluate_method(method, image, gt_mask)
            except Exception as e:
                print(f"  [ERROR] {method} failed on {img_path.name}: {e}")
                iou = 0.0
                choice = "error"

            row[f"{method}_iou"] = iou
            row[f"{method}_choice"] = choice
            method_ious[method].append(iou)

        results.append(row)

    csv_path = OUTPUT_DIR / "baseline_iou_results.csv"

    fieldnames = ["image", "mask"]

    for method in methods:
        fieldnames.append(f"{method}_iou")
        fieldnames.append(f"{method}_choice")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n=== SUMMARY ===")

    for method in methods:
        stats = summarize(method_ious[method])
        print(
            f"{method:12s} | "
            f"mean={stats['mean']:.3f} | "
            f"median={stats['median']:.3f} | "
            f"std={stats['std']:.3f} | "
            f"min={stats['min']:.3f} | "
            f"max={stats['max']:.3f}"
        )

    print(f"\nSaved detailed results to: {csv_path}")


if __name__ == "__main__":
    main()