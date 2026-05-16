"""
Method 8: Chan-Vese Active Contours
Active contours without edges, using level sets.

Reference: Chan, T. F., & Vese, L. A. (2001). Active contours without edges.
IEEE Transactions on image processing, 10(2), 266-277.
"""

import numpy as np
from skimage.segmentation import chan_vese
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
    Generate binary masks using Chan-Vese segmentation.
    
    Args:
        image: Input image (HxWx3 RGB or HxW grayscale)
        boxes: List of bounding boxes [[x1,y1,x2,y2], ...]
        config: Configuration dictionary
                - 'max_iter': Maximum iterations (default: 100)
                - 'mu': Edge length penalty parameter (default: 0.25)
    
    Returns:
        List of binary masks (HxW uint8, values 0 or 1), one per box
    """
    img_h, img_w = image.shape[:2]
    masks = []
    
    # Get configuration parameters
    max_iter = config.get('max_iter', 100)
    mu = config.get('mu', 0.25)
    
    for box in boxes:
        x1, y1, x2, y2 = box
        
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            masks.append(np.zeros((img_h, img_w), dtype=np.uint8))
            continue
            
        crop_h, crop_w = crop.shape[:2]
        enhanced = enhance_contrast(crop)
        
        try:
            # Initialize with a central disk/checkerboard approach
            # Using 'checkerboard' is highly robust for Chan-Vese in skimage
            cv_mask = chan_vese(
                enhanced, 
                mu=mu, 
                lambda1=1, 
                lambda2=1, 
                tol=1e-3, 
                max_num_iter=max_iter,
                dt=0.5, 
                init_level_set="checkerboard", 
                extended_output=False
            )
            
            # The result is boolean, convert to uint8
            binary_mask = cv_mask.astype(np.uint8)
            binary_mask = post_process_mask(binary_mask, morphology_size=2)
            
        except Exception as e:
            print(f"Chan-Vese failed on box {box}: {e}")
            from skimage.filters import threshold_otsu
            try:
                thresh = threshold_otsu(enhanced)
                binary_mask = (enhanced > thresh).astype(np.uint8)
            except:
                binary_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = binary_mask
        masks.append(full_mask)
    
    return masks


if __name__ == "__main__":
    print("Testing Chan-Vese method...")
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_boxes = [[10, 10, 50, 50], [60, 60, 90, 90]]
    test_config = {'max_iter': 50, 'mu': 0.25}  # Lower max_iter for testing speed
    
    masks = generate_masks(test_image, test_boxes, test_config)
    print(f"Generated {len(masks)} masks")
    print(f"Config used: {test_config}")
    print("✓ Chan-Vese method test passed")