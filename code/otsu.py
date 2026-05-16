"""
Method 1: Otsu Thresholding
Single global threshold based on histogram analysis.

Reference: Otsu, N. (1979). A threshold selection method from gray-level histograms.
IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62-66.
"""

import numpy as np
from skimage.filters import threshold_otsu
from typing import List, Dict
import sys
import os
sys.path.append(os.path.dirname(__file__))
from utils import enhance_contrast, post_process_mask


def generate_masks(
    image: np.ndarray,
    boxes: List[List[int]],
    config: Dict
) -> List[np.ndarray]:
    """
    Generate binary masks using Otsu thresholding.
    
    Args:
        image: Input image (HxWx3 RGB or HxW grayscale)
        boxes: List of bounding boxes [[x1,y1,x2,y2], ...]
        config: Configuration dictionary (not used for Otsu, but required by signature)
    
    Returns:
        List of binary masks (HxW uint8, values 0 or 1), one per box
    """
    img_h, img_w = image.shape[:2]
    masks = []
    
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # Extract crop
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            # Empty crop - return empty mask
            masks.append(np.zeros((img_h, img_w), dtype=np.uint8))
            continue
        
        # Enhance contrast
        enhanced = enhance_contrast(crop)
        
        # Apply Otsu thresholding
        try:
            thresh_value = threshold_otsu(enhanced)
            binary_mask = (enhanced > thresh_value).astype(np.uint8)
        except Exception as e:
            print(f"Otsu failed on box {box}: {e}")
            binary_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        
        # Post-process: fill holes and clean up
        binary_mask = post_process_mask(binary_mask, morphology_size=2)
        
        # Create full-size mask
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = binary_mask
        
        masks.append(full_mask)
    
    return masks


if __name__ == "__main__":
    # Test the function
    print("Testing Otsu thresholding method...")
    
    # Create a simple test image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_boxes = [[10, 10, 50, 50], [60, 60, 90, 90]]
    test_config = {}
    
    masks = generate_masks(test_image, test_boxes, test_config)
    
    print(f"Generated {len(masks)} masks")
    print(f"Mask shapes: {[m.shape for m in masks]}")
    print(f"Mask dtypes: {[m.dtype for m in masks]}")
    print(f"Mask value ranges: {[(m.min(), m.max()) for m in masks]}")
    print("✓ Otsu method test passed")