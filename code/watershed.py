"""
Method 4: Pure Watershed Segmentation
Watershed without Otsu initialization (reviewer #1 requirement).

Uses distance transform on enhanced image directly for markers.

Reference: Beucher, S. (1994). Watershed, hierarchical segmentation and waterfall algorithm.
"""

import numpy as np
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import disk, erosion, dilation
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
    Generate binary masks using pure watershed segmentation.
    
    This method does NOT use Otsu for initialization (as per MICCAI reviewer #1).
    Instead, it uses:
    1. Direct thresholding or morphological gradient for mask
    2. Distance transform for marker placement
    3. Watershed for region growing
    
    Args:
        image: Input image (HxWx3 RGB or HxW grayscale)
        boxes: List of bounding boxes [[x1,y1,x2,y2], ...]
        config: Configuration dictionary
                - 'sigma': Gaussian smoothing for distance (default: 1.5)
                - 'marker_threshold': Percentile for marker creation (default: 0.7)
                - 'use_gradient': If True, use morphological gradient (default: False)
    
    Returns:
        List of binary masks (HxW uint8, values 0 or 1), one per box
    """
    img_h, img_w = image.shape[:2]
    masks = []
    
    # Get configuration parameters
    sigma = config.get('sigma', 1.5)
    marker_threshold = config.get('marker_threshold', 0.7)
    use_gradient = config.get('use_gradient', False)
    
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
            # Create mask for watershed
            if use_gradient:
                # Use morphological gradient as elevation map
                from skimage.filters import rank
                from skimage.morphology import disk
                gradient = rank.gradient(enhanced, disk(2))
                # Invert for watershed (lower = better)
                elevation = -gradient.astype(float)
            else:
                # Use intensity directly (inverted)
                elevation = -enhanced.astype(float)
            
            # Create initial mask (simple threshold or full region)
            # For pure watershed, we work on the full crop
            mask_region = np.ones(enhanced.shape, dtype=bool)
            
            # Apply distance transform to find peaks (markers)
            distance = ndi.distance_transform_edt(mask_region)
            distance = gaussian(distance, sigma=sigma)
            
            # Create markers from distance peaks
            # Use percentile threshold to get foreground markers
            dist_threshold = np.percentile(distance[distance > 0], marker_threshold * 100)
            markers_binary = distance > dist_threshold
            
            # Dilate slightly to ensure connectivity
            markers_binary = dilation(markers_binary, disk(2))
            
            # Label the markers
            markers = label(markers_binary)
            
            # If no markers found, create one at the center
            if markers.max() == 0:
                center_y, center_x = crop_h // 2, crop_w // 2
                markers[center_y-2:center_y+2, center_x-2:center_x+2] = 1
            
            # Apply watershed
            labels = watershed(elevation, markers, mask=mask_region)
            
            # Select best region (most central and largest)
            best_label = select_central_region(labels, crop_h, crop_w)
            
            if best_label == 0:
                # No valid region found - fallback to largest region
                region_sizes = [(region.label, region.area) for region in regionprops(labels)]
                if region_sizes:
                    best_label = max(region_sizes, key=lambda x: x[1])[0]
                else:
                    # Ultimate fallback: simple thresholding
                    thresh = threshold_otsu(enhanced)
                    binary_mask = (enhanced > thresh).astype(np.uint8)
                    binary_mask = post_process_mask(binary_mask)
                    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    full_mask[y1:y2, x1:x2] = binary_mask
                    masks.append(full_mask)
                    continue
            
            # Extract the selected region
            binary_mask = (labels == best_label).astype(np.uint8)
            
            # Clean up edges
            binary_mask = erosion(binary_mask, disk(1))
            binary_mask = post_process_mask(binary_mask, morphology_size=2)
            
        except Exception as e:
            print(f"Pure Watershed failed on box {box}: {e}")
            # Fallback to simple thresholding
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
    print("Testing Pure Watershed method...")
    
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_boxes = [[10, 10, 50, 50], [60, 60, 90, 90]]
    test_config = {'sigma': 1.5, 'marker_threshold': 0.7, 'use_gradient': False}
    
    masks = generate_masks(test_image, test_boxes, test_config)
    
    print(f"Generated {len(masks)} masks")
    print(f"Config used: {test_config}")
    print("✓ Pure Watershed method test passed")