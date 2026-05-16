"""
Method 2: Multi-Otsu Thresholding
Three-class thresholding for more refined segmentation.

Reference: Liao, P. S., Chen, T. S., & Chung, P. C. (2001).
A fast algorithm for multilevel thresholding.
Journal of Information Science and Engineering, 17(5), 713-727.
"""

import numpy as np
from skimage.filters import threshold_multiotsu
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
    Generate binary masks using Multi-Otsu thresholding (3 classes).
    
    Args:
        image: Input image (HxWx3 RGB or HxW grayscale)
        boxes: List of bounding boxes [[x1,y1,x2,y2], ...]
        config: Configuration dictionary
                - 'classes': Number of classes (default: 3)
                - 'use_highest_class': If True, use highest intensity class (default: True)
    
    Returns:
        List of binary masks (HxW uint8, values 0 or 1), one per box
    """
    img_h, img_w = image.shape[:2]
    masks = []
    
    # Get configuration parameters
    n_classes = config.get('classes', 3)
    use_highest = config.get('use_highest_class', True)
    
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # Extract crop
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            masks.append(np.zeros((img_h, img_w), dtype=np.uint8))
            continue
        
        # Enhance contrast
        enhanced = enhance_contrast(crop)
        
        # Apply Multi-Otsu thresholding
        try:
            thresholds = threshold_multiotsu(enhanced, classes=n_classes)
            
            # Use the highest intensity class (foreground)
            if use_highest:
                binary_mask = (enhanced > thresholds[-1]).astype(np.uint8)
            else:
                # Use middle threshold (good for 3-class)
                binary_mask = (enhanced > thresholds[len(thresholds)//2]).astype(np.uint8)
                
        except Exception as e:
            print(f"Multi-Otsu failed on box {box}: {e}")
            # Fallback to simple thresholding
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
    print("Testing Multi-Otsu thresholding method...")
    
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_boxes = [[10, 10, 50, 50], [60, 60, 90, 90]]
    test_config = {'classes': 3, 'use_highest_class': True}
    
    masks = generate_masks(test_image, test_boxes, test_config)
    
    print(f"Generated {len(masks)} masks")
    print(f"Config used: {test_config}")
    print("✓ Multi-Otsu method test passed")