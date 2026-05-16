"""
Method 5: Otsu + Watershed (Reference Baseline)
Combines Otsu thresholding with watershed segmentation.

This is the reference baseline method from the original MICCAI paper.
Otsu provides initial markers, watershed refines boundaries.

References:
- Otsu, N. (1979). A threshold selection method from gray-level histograms.
- Beucher, S. (1994). Watershed, hierarchical segmentation and waterfall algorithm.
"""

import numpy as np
from skimage.filters import threshold_otsu, threshold_multiotsu, gaussian
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage.morphology import disk, erosion
from scipy import ndimage as ndi
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
    Generate binary masks using Otsu + Watershed segmentation.
    
    This is the reference baseline method that combines:
    1. Otsu thresholding for initial loose mask
    2. Multi-Otsu for strict markers
    3. Distance transform for watershed seeds
    4. Watershed for final segmentation
    
    Args:
        image: Input image (HxWx3 RGB or HxW grayscale)
        boxes: List of bounding boxes [[x1,y1,x2,y2], ...]
        config: Configuration dictionary
                - 'sigma': Gaussian smoothing for distance transform (default: 1.0)
                - 'erosion_size': Size for marker erosion (default: 5)
    
    Returns:
        List of binary masks (HxW uint8, values 0 or 1), one per box
    """
    img_h, img_w = image.shape[:2]
    masks = []
    
    # Get configuration parameters
    sigma = config.get('sigma', 1.0)
    erosion_size = config.get('erosion_size', 5)
    
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
            # Step 1: Otsu thresholding for loose mask
            thresh_otsu = threshold_otsu(enhanced)
            mask_loose = enhanced > thresh_otsu
            mask_loose = ndi.binary_fill_holes(mask_loose)
            mask_loose = post_process_mask(mask_loose, morphology_size=3)
            
            # Step 2: Distance transform
            distance = ndi.distance_transform_edt(mask_loose)
            distance = gaussian(distance, sigma=sigma)
            
            # Step 3: Multi-Otsu for strict markers (foreground seeds)
            try:
                thresholds = threshold_multiotsu(enhanced, classes=3)
                mask_strict = enhanced > thresholds[1]
                mask_strict = ndi.binary_fill_holes(mask_strict)
                
                # If strict mask is empty, use erosion of loose mask
                if np.sum(mask_strict) == 0:
                    mask_strict = erosion(mask_loose, disk(erosion_size))
            except:
                # Fallback: erode loose mask
                mask_strict = erosion(mask_loose, disk(erosion_size))
            
            # Step 4: Create markers from strict mask
            markers = label(mask_strict)
            
            # Step 5: Watershed segmentation
            labels = watershed(-distance, markers, mask=mask_loose)
            
            # Step 6: Select best region (most central and largest)
            best_label = select_central_region(labels, crop_h, crop_w)
            
            if best_label == 0:
                # No valid region found
                binary_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
            else:
                # Extract the selected region
                binary_mask = (labels == best_label).astype(np.uint8)
                # Final erosion to tighten boundaries slightly
                binary_mask = erosion(binary_mask, disk(1))
            
        except Exception as e:
            print(f"Otsu+Watershed failed on box {box}: {e}")
            # Fallback to simple Otsu
            try:
                thresh = threshold_otsu(enhanced)
                binary_mask = (enhanced > thresh).astype(np.uint8)
                binary_mask = post_process_mask(binary_mask)
            except:
                binary_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        
        # Create full-size mask
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = binary_mask
        
        masks.append(full_mask)
    
    return masks


if __name__ == "__main__":
    # Test the function
    print("Testing Otsu + Watershed (Reference Baseline) method...")
    
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_boxes = [[10, 10, 50, 50], [60, 60, 90, 90]]
    test_config = {'sigma': 1.0, 'erosion_size': 5}
    
    masks = generate_masks(test_image, test_boxes, test_config)
    
    print(f"Generated {len(masks)} masks")
    print(f"Config used: {test_config}")
    print("✓ Otsu+Watershed method test passed")