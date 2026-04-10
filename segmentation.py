"""
segmentation.py — AI Banana Preprocessor
============================================
Goal: Make ANY banana image look like training data
      (white background, banana centered, 224x224)

Pipeline:
    Any Image (any background)
         ↓
    Step 1: Deep Learning Background Removal (rembg U^2-Net)
         ↓
    Step 2: Calculate Bounding Box on Foreground Mask
         ↓
    Step 3: Crop to banana + padding
         ↓
    Step 4: Place on pure white background
         ↓
    224×224 white-background image → CNN ✅
"""

import cv2
import numpy as np
import os
from PIL import Image
from rembg import remove

def get_banana_bbox(mask, padding_pct=0.12):
    """Find bounding box of banana with padding from an alpha mask."""
    # mask is a 2D numpy array containing 0-255 alpha values
    # find where alpha > 0
    y_indices, x_indices = np.where(mask > 0)
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None
        
    x = np.min(x_indices)
    y = np.min(y_indices)
    w = np.max(x_indices) - x
    h = np.max(y_indices) - y
    
    H, W = mask.shape
    px = int(w * padding_pct)
    py = int(h * padding_pct)
    return (max(0, x-px), max(0, y-py),
            min(W, x+w+px), min(H, y+h+py))

# ══════════════════════════════════════════════════════════════
#  MAIN FUNCTION
# ══════════════════════════════════════════════════════════════

def extract_banana_roi(input_path: str,
                       output_path: str,
                       target_size: tuple = (224, 224),
                       debug: bool = False) -> dict:
    """
    Converts any banana image to white-background version
    that matches CNN training data format using U^2-Net AI.
    """
    result = {"success": False, "output_path": output_path,
              "method": "none", "message": ""}

    # ── Load & resize for processing ──────────────────────────
    try:
        # Load image via PIL to guarantee RGB
        img_pil = Image.open(input_path).convert("RGB")
        # Resize to speed up U^2-Net processing, but keeping aspect ratio or just 512x512
        img_pil = img_pil.resize((512, 512))
    except Exception as e:
        result["message"] = f"Load failed: {e}"
        return result

    # ── Step 1: AI Background Removal (rembg) ──────────────────
    try:
        # returns an RGBA image where background is transparent
        output_rgba = remove(img_pil)
        out_np = np.array(output_rgba)
        
        # alpha channel is the mask
        alpha_mask = out_np[:, :, 3]
        
        banana_pct = np.sum(alpha_mask > 0) / (512 * 512)
        
        if banana_pct > 0.01:  # At least 1% of image is banana
            # ── Step 2: Crop to banana ─────────────────────────
            bbox = get_banana_bbox(alpha_mask)
            if bbox:
                x1, y1, x2, y2 = bbox
                cropped_rgba = out_np[y1:y2, x1:x2]
                
                if cropped_rgba.shape[0] > 20 and cropped_rgba.shape[1] > 20:
                    # ── Step 3: White background ──────────────────────
                    # Paste the RGBA onto a pure white background
                    white_bg = np.ones((cropped_rgba.shape[0], cropped_rgba.shape[1], 3), dtype=np.uint8) * 255
                    
                    # Alpha compositing using the alpha channel
                    alpha = cropped_rgba[:, :, 3] / 255.0
                    for c in range(3):
                        white_bg[:, :, c] = (alpha * cropped_rgba[:, :, c] +
                                             (1.0 - alpha) * white_bg[:, :, c])
                                             
                    # Resize to target
                    final = cv2.resize(white_bg, target_size)
                    cv2.imwrite(output_path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
                    
                    result.update({
                        "success": True,
                        "method":  "rembg_u2net+crop",
                        "message": f"AI segmentation successful ({banana_pct*100:.1f}% coverage). "
                                   f"Background replaced with white."
                    })
                    
                    if debug:
                        base = os.path.splitext(output_path)[0]
                        cv2.imwrite(f"{base}_mask.jpg", alpha_mask)
                        cv2.imwrite(f"{base}_white.jpg", cv2.cvtColor(white_bg, cv2.COLOR_RGB2BGR))
                        
                    return result
        else:
            # If nothing was detected
            result["message"] = "AI could not detect any salient foreground object."
            pass

    except Exception as e:
        result["message"] = f"Segmentation failed: {e}"

    # ── Fallback: just resize ─────────────────────────────────
    try:
        img_bgr = cv2.imread(input_path)
        if img_bgr is None:
            img_bgr = cv2.cvtColor(np.array(Image.open(input_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        final = cv2.resize(img_bgr, target_size)
        cv2.imwrite(output_path, final)
        result.update({"success": True, "method": "fallback",
                       "message": result["message"] + " Used original image."})
    except Exception as e:
        result["message"] += f" Fallback failed: {e}"

    return result

# ── Test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        r = extract_banana_roi(sys.argv[1],
                               sys.argv[1].replace(".", "_processed."),
                               debug=True)
        print(f"Success : {r['success']}")
        print(f"Method  : {r['method']}")
        print(f"Message : {r['message']}")
    else:
        print("Usage: python segmentation.py <image_path>")
