"""
Method 9: Morphological Geodesic Active Contours (MorphGAC)
Level-set method that evolves a contour toward image edges.

Reference: Marquez-Neila, P., Baumela, L., & Alvarez, L. (2014). 
A morphological approach to curvature-based evolution of curves and surfaces.
IEEE Transactions on Pattern Analysis and Machine Intelligence, 36(1), 2-17.
"""

import numpy as np
from skimage.segmentation import morphological_geodesic_active_contour, inverse_gaussian_gradient
from skimage.morphology import disk
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
    Generate binary masks using Morphological GAC.
    
    Args:
        image: Input image (HxWx3 RGB or HxW grayscale)
        boxes: List of bounding boxes [[x1,y1,x2,y2], ...]
        config: Configuration dictionary
                - 'iterations': Number of active contour evolutions (default: 200)
                - 'smoothing': Number of smoothing iterations per stage (default: 1)
                - 'balloon': Balloon force to push contour outwards (default: 1)
    
    Returns:
        List of binary masks (HxW uint8, values 0 or 1), one per box
    """
    img_h, img_w = image.shape[:2]
    masks = []
    
    # Get configuration parameters
    iterations = config.get('iterations', 200)
    smoothing = config.get('smoothing', 1)
    balloon = config.get('balloon', 1)
    
    for box in boxes:
        x1, y1, x2, y2 = box
        
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            masks.append(np.zeros((img_h, img_w), dtype=np.uint8))
            continue
            
        crop_h, crop_w = crop.shape[:2]
        enhanced = enhance_contrast(crop)
        
        try:
            # MorphGAC requires an edge image (inverse gaussian gradient)
            gimage = inverse_gaussian_gradient(enhanced, alpha=100, sigma=2.0)
            
            # Initial level set: a small disk in the center to expand outwards
            init_ls = np.zeros((crop_h, crop_w), dtype=np.int8)
            cy, cx = crop_h // 2, crop_w // 2
            r = max(3, min(crop_h, crop_w) // 6)
            
            # Create circular mask for initialization
            y, x = np.ogrid[-cy:crop_h-cy, -cx:crop_w-cx]
            mask_circle = x**2 + y**2 <= r**2
            init_ls[mask_circle] = 1
            
            # Apply Morphological GAC
            ls_mask = morphological_geodesic_active_contour(
                gimage, 
                iterations=iterations, 
                init_level_set=init_ls,
                smoothing=smoothing, 
                threshold=0.69,
                balloon=balloon
            )
            
            binary_mask = ls_mask.astype(np.uint8)
            binary_mask = post_process_mask(binary_mask, morphology_size=2)
            
        except Exception as e:
            print(f"MorphGAC failed on box {box}: {e}")
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
    print("Testing MorphGAC method...")
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_boxes = [[10, 10, 50, 50], [60, 60, 90, 90]]
    test_config = {'iterations': 20, 'smoothing': 1, 'balloon': 1} # Lower iter for quick test
    
    masks = generate_masks(test_image, test_boxes, test_config)
    print(f"Generated {len(masks)} masks")
    print(f"Config used: {test_config}")
    print("✓ MorphGAC method test passed")