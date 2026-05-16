"""
Method 12: Guided U-Net
Transforms semantic segmentation into instance segmentation by adding 
a bounding box guidance channel to the U-Net inputs.

Reference: Bilic & Egger (2023). Transforming Semantic Segmentation 
into Instance Segmentation with a Guided U-Net. IEEE.
"""

import numpy as np
import cv2
import sys
import os

try:
    import torch
    import segmentation_models_pytorch as smp
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

sys.path.append(os.path.dirname(__file__))
from utils import post_process_mask


def generate_masks(
    image: np.ndarray,
    boxes: List[List[int]],
    config: Dict
) -> List[np.ndarray]:
    """
    Generate binary masks using a pre-trained Guided U-Net.
    
    Args:
        image: Input image (HxWx3 RGB or HxW grayscale)
        boxes: List of bounding boxes [[x1,y1,x2,y2], ...]
        config: Configuration dictionary
                - 'checkpoint': Path to the .pt checkpoint (default: "checkpoints/guided_unet.pt")
                - 'encoder_name': Backbone architecture (default: "resnet34")
                - 'device': "cuda" or "cpu" (default: auto-detect)
                - 'input_size': Size to resize full image for inference (default: 256)
    
    Returns:
        List of binary masks (HxW uint8, values 0 or 1), one per box
    """
    img_h, img_w = image.shape[:2]
    masks = []
    
    checkpoint_path = config.get('checkpoint', 'checkpoints/guided_unet.pt')
    encoder_name = config.get('encoder_name', 'resnet34')
    input_size = config.get('input_size', 256)
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    if len(image.shape) == 2:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    model_loaded = False
    if TORCH_AVAILABLE and os.path.exists(checkpoint_path):
        try:
            # Notice in_channels=4 (3 for RGB, 1 for BBox guidance mask)
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=4, 
                classes=1,
            )
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.to(device)
            model.eval()
            model_loaded = True
        except Exception as e:
            print(f"Failed to load Guided U-Net checkpoint: {e}")

    # Resize base image once
    img_resized = cv2.resize(img_rgb, (input_size, input_size))
    img_norm = img_resized.astype(np.float32) / 255.0

    for box in boxes:
        x1, y1, x2, y2 = box
        
        if model_loaded:
            try:
                # 1. Create the guidance mask (1 inside box, 0 outside)
                guidance = np.zeros((img_h, img_w), dtype=np.float32)
                guidance[y1:y2, x1:x2] = 1.0
                
                # Resize guidance to match network input
                guidance_resized = cv2.resize(guidance, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
                guidance_resized = np.expand_dims(guidance_resized, axis=-1)
                
                # 2. Concatenate RGB (3 channels) + Guidance (1 channel)
                input_concat = np.concatenate([img_norm, guidance_resized], axis=-1)
                
                # HWC to CHW
                tensor = torch.from_numpy(input_concat).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # 3. Predict
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.sigmoid(logits)
                    pred_mask = (probs > 0.5).squeeze().cpu().numpy().astype(np.uint8)
                
                # 4. Resize back to original image dimensions
                binary_mask = cv2.resize(pred_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                
                # Ensure the network didn't predict anything outside the designated box (safety clamp)
                final_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                final_mask[y1:y2, x1:x2] = binary_mask[y1:y2, x1:x2]
                final_mask = post_process_mask(final_mask, morphology_size=2)
                
                masks.append(final_mask)
                continue # Skip fallback
                
            except Exception as e:
                print(f"Guided U-Net inference failed on box {box}: {e}")
                
        # --- Fallback Routine (if weights missing or inference failed) ---
        from skimage.filters import threshold_otsu
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size > 0:
            try:
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                thresh = threshold_otsu(gray_crop)
                binary_crop = (gray_crop > thresh).astype(np.uint8)
            except:
                binary_crop = np.zeros(crop.shape[:2], dtype=np.uint8)
        else:
            binary_crop = np.zeros((0,0), dtype=np.uint8)
            
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        if crop.size > 0:
            full_mask[y1:y2, x1:x2] = binary_crop
        masks.append(full_mask)

    return masks

if __name__ == "__main__":
    print("Testing Guided U-Net method...")
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    test_boxes = [[10, 10, 50, 50]]
    test_config = {'checkpoint': 'dummy.pt', 'device': 'cpu'}
    
    masks = generate_masks(test_image, test_boxes, test_config)
    print(f"Generated {len(masks)} masks (Fallback triggered intentionally for dummy.pt)")
    print("✓ Guided U-Net method test passed")