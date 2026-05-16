"""
Method 7: Random Walker
Probabilistic image segmentation given seed points. 
Since we use bounding boxes, the center is assumed foreground and borders are background.

Reference: Grady, L. (2006). Random walks for image segmentation. 
IEEE Transactions on Pattern Analysis and Machine Intelligence, 28(11), 1768-1783.
"""

import numpy as np
from skimage.segmentation import random_walker
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
    Generate binary masks using Random Walker segmentation.
    
    Args:
        image: Input image (HxWx3 RGB or HxW grayscale)
        boxes: List of bounding boxes [[x1,y1,x2,y2], ...]
        config: Configuration dictionary
                - 'beta': Penalization weight for local changes (default: 130)
    
    Returns:
        List of binary masks (HxW uint8, values 0 or 1), one per box
    """
    img_h, img_w = image.shape[:2]
    masks = []
    
    # Get configuration parameters
    beta = config.get('beta', 130)
    
    for box in boxes:
        x1, y1, x2, y2 = box
        
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            masks.append(np.zeros((img_h, img_w), dtype=np.uint8))
            continue
            
        crop_h, crop_w = crop.shape[:2]
        enhanced = enhance_contrast(crop)
        
        try:
            # Create markers map (0: unlabelled, 1: foreground, 2: background)
            markers = np.zeros((crop_h, crop_w), dtype=np.uint8)
            
            # Seed background (edges of the bounding box)
            markers[0, :] = 2
            markers[-1, :] = 2
            markers[:, 0] = 2
            markers[:, -1] = 2
            
            # Seed foreground (a small box in the direct center)
            cy, cx = crop_h // 2, crop_w // 2
            r = max(2, min(crop_h, crop_w) // 8) # Central radius
            markers[cy-r:cy+r, cx-r:cx+r] = 1
            
            # Apply Random Walker
            labels = random_walker(enhanced, markers, beta=beta, mode='bf')
            
            # Extract foreground (label 1)
            binary_mask = (labels == 1).astype(np.uint8)
            binary_mask = post_process_mask(binary_mask, morphology_size=2)
            
        except Exception as e:
            print(f"Random Walker failed on box {box}: {e}")
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
    print("Testing Random Walker method...")
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_boxes = [[10, 10, 50, 50], [60, 60, 90, 90]]
    test_config = {'beta': 130}
    
    masks = generate_masks(test_image, test_boxes, test_config)
    print(f"Generated {len(masks)} masks")
    print(f"Config used: {test_config}")
    print("✓ Random Walker method test passed")