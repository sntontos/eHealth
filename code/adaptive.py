"""
Method 3: Adaptive (Local) Thresholding
Local thresholding that adapts to image regions.

Uses OpenCV's adaptiveThreshold with Gaussian weighting.
"""

import numpy as np
import cv2
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
    Generate binary masks using adaptive thresholding.
    
    Args:
        image: Input image (HxWx3 RGB or HxW grayscale)
        boxes: List of bounding boxes [[x1,y1,x2,y2], ...]
        config: Configuration dictionary
                - 'block_size': Size of pixel neighborhood (default: 35, must be odd)
                - 'C': Constant subtracted from weighted mean (default: 2)
                - 'method': 'gaussian' or 'mean' (default: 'gaussian')
    
    Returns:
        List of binary masks (HxW uint8, values 0 or 1), one per box
    """
    img_h, img_w = image.shape[:2]
    masks = []
    
    # Get configuration parameters
    block_size = config.get('block_size', 35)
    C = config.get('C', 2)
    method = config.get('method', 'gaussian')
    
    # Ensure block_size is odd
    if block_size % 2 == 0:
        block_size += 1
    
    # Select adaptive method
    if method == 'mean':
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
    else:
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # Extract crop
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            masks.append(np.zeros((img_h, img_w), dtype=np.uint8))
            continue
        
        # Enhance contrast
        enhanced = enhance_contrast(crop)
        
        # Ensure minimum size for adaptive threshold
        crop_h, crop_w = enhanced.shape[:2]
        if crop_h < block_size or crop_w < block_size:
            # Fallback to simple thresholding for very small crops
            mean_val = np.mean(enhanced)
            binary_mask = (enhanced > mean_val).astype(np.uint8)
        else:
            # Apply adaptive thresholding
            try:
                binary_mask = cv2.adaptiveThreshold(
                    enhanced,
                    maxValue=1,  # Output values will be 0 or 1
                    adaptiveMethod=adaptive_method,
                    thresholdType=cv2.THRESH_BINARY,
                    blockSize=block_size,
                    C=C
                )
            except Exception as e:
                print(f"Adaptive threshold failed on box {box}: {e}")
                mean_val = np.mean(enhanced)
                binary_mask = (enhanced > mean_val).astype(np.uint8)
        
        # Post-process
        binary_mask = post_process_mask(binary_mask, morphology_size=2)
        
        # Create full-size mask
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = binary_mask
        
        masks.append(full_mask)
    
    return masks


if __name__ == "__main__":
    # Test the function
    print("Testing Adaptive thresholding method...")
    
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_boxes = [[10, 10, 50, 50], [60, 60, 90, 90]]
    test_config = {'block_size': 35, 'C': 2, 'method': 'gaussian'}
    
    masks = generate_masks(test_image, test_boxes, test_config)
    
    print(f"Generated {len(masks)} masks")
    print(f"Config used: {test_config}")
    print("✓ Adaptive threshold method test passed")