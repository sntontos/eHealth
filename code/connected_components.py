"""
Method 6: Connected Components
Segmentation using cv2.connectedComponents after simple binarization.
This method isolates the main object by removing disconnected noise.

Requirement from MICCAI Reviewer #1.
"""

import numpy as np
import cv2
from skimage.filters import threshold_otsu
from typing import List, Dict
import sys
import os
sys.path.append(os.path.dirname(__file__))
from utils import enhance_contrast, post_process_mask, select_central_region


def generate_masks(
    image: np.ndarray,
    boxes: List[List[int]],
    config: Dict
) -> List[np.ndarray]:
    """
    Generate binary masks using Connected Components after binarization.
    
    Args:
        image: Input image (HxWx3 RGB or HxW grayscale)
        boxes: List of bounding boxes [[x1,y1,x2,y2], ...]
        config: Configuration dictionary
                - 'connectivity': 4 or 8 (default: 8)
    
    Returns:
        List of binary masks (HxW uint8, values 0 or 1), one per box
    """
    img_h, img_w = image.shape[:2]
    masks = []
    
    # Get configuration parameters
    connectivity = config.get('connectivity', 8)
    
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # Extract crop
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            masks.append(np.zeros((img_h, img_w), dtype=np.uint8))
            continue
        
        crop_h, crop_w = crop.shape[:2]
        
        # Enhance contrast
        enhanced = enhance_contrast(crop)
        
        try:
            # 1. Simple binarization (Otsu)
            thresh_value = threshold_otsu(enhanced)
            binary_base = (enhanced > thresh_value).astype(np.uint8)
            
            # 2. Connected Components
            num_labels, labels = cv2.connectedComponents(binary_base, connectivity=connectivity)
            
            if num_labels <= 1:
                # No foreground found
                binary_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
            else:
                # 3. Select the best region (central and largest)
                best_label = select_central_region(labels, crop_h, crop_w)
                
                if best_label == 0:
                    binary_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
                else:
                    binary_mask = (labels == best_label).astype(np.uint8)
                    binary_mask = post_process_mask(binary_mask, morphology_size=2)
                    
        except Exception as e:
            print(f"Connected Components failed on box {box}: {e}")
            # Fallback to simple Otsu
            try:
                thresh_value = threshold_otsu(enhanced)
                binary_mask = (enhanced > thresh_value).astype(np.uint8)
            except:
                binary_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        
        # Create full-size mask
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = binary_mask
        
        masks.append(full_mask)
    
    return masks


if __name__ == "__main__":
    print("Testing Connected Components method...")
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_boxes = [[10, 10, 50, 50], [60, 60, 90, 90]]
    test_config = {'connectivity': 8}
    
    masks = generate_masks(test_image, test_boxes, test_config)
    print(f"Generated {len(masks)} masks")
    print(f"Config used: {test_config}")
    print("✓ Connected Components method test passed")