"""
segmentation.py — Smart Banana Preprocessor
============================================
Key fix: All images saved as RGB (PIL) — NOT BGR (OpenCV)
This matches exactly what Keras ImageDataGenerator expects.

Pipeline:
    Any Image (any background)
         ↓
    Color detection (HSV + LAB)
         ↓
    GrabCut refinement
         ↓
    White background replacement
         ↓
    Crop to banana
         ↓
    Save as RGB PNG → CNN ✅
"""

import cv2
import numpy as np
import os
from PIL import Image


def load_image_rgb(path):
    """Load image as RGB numpy array."""
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except Exception:
        img_bgr = cv2.imread(path)
        if img_bgr is not None:
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        raise ValueError(f"Cannot load image: {path}")


def detect_banana_mask(img_rgb):
    """
    Detects banana pixels using HSV + LAB color spaces.
    Input: RGB image
    Returns: binary mask (255=banana, 0=background)
    """
    # Convert to HSV and LAB for detection
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    # Green banana (unripe)
    m1 = cv2.inRange(hsv, np.array([25, 30, 30]),  np.array([90, 255, 255]))
    # Yellow banana (ripe)
    m2 = cv2.inRange(hsv, np.array([15, 50, 80]),  np.array([40, 255, 255]))
    # Dark yellow (overripe)
    m3 = cv2.inRange(hsv, np.array([10, 30, 40]),  np.array([25, 220, 200]))
    # Brown/black (rotten)
    m4 = cv2.inRange(hsv, np.array([0,  20, 10]),  np.array([20, 180, 140]))
    # Very dark banana
    m5 = cv2.inRange(hsv, np.array([20, 10, 10]),  np.array([35, 100, 100]))
    # LAB b-channel (yellow/green detection)
    b_ch = lab[:, :, 2]
    m6   = np.where(b_ch > 140, 255, 0).astype(np.uint8)

    # Combine
    combined = cv2.bitwise_or(m1, m2)
    combined = cv2.bitwise_or(combined, m3)
    combined = cv2.bitwise_or(combined, m4)
    combined = cv2.bitwise_or(combined, m5)
    combined = cv2.bitwise_or(combined, m6)

    # Morphological cleanup
    k_large  = np.ones((15, 15), np.uint8)
    k_small  = np.ones((5,  5),  np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_large, iterations=4)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k_small, iterations=2)
    combined = cv2.dilate(combined, k_small, iterations=3)

    return combined


def grabcut_refine(img_rgb, initial_mask):
    """
    Refine mask using GrabCut initialized with color mask.
    Input/output: RGB image
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w    = img_bgr.shape[:2]

    gc_mask          = np.zeros((h, w), np.uint8)
    gc_mask[:]       = cv2.GC_BGD
    gc_mask[initial_mask > 0] = cv2.GC_PR_FGD

    # Mark center of largest contour as definite foreground
    contours, _ = cv2.findContours(initial_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest  = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        cx, cy   = x + bw // 2, y + bh // 2
        r        = min(bw, bh) // 4
        cv2.circle(gc_mask, (cx, cy), r, cv2.GC_FGD, -1)

    try:
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(img_bgr, gc_mask, None, bgd, fgd, 5,
                    cv2.GC_INIT_WITH_MASK)
        refined = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
            255, 0
        ).astype(np.uint8)

        if np.sum(refined > 0) > 1000:
            k       = np.ones((7, 7), np.uint8)
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, k, iterations=3)
            return refined
    except Exception:
        pass

    return initial_mask


def make_white_background_rgb(img_rgb, mask):
    """
    Replace background with pure white.
    Input/Output: RGB images — NO BGR conversion!
    This is critical — CNN was trained on RGB images.
    """
    white   = np.ones_like(img_rgb, dtype=np.float32) * 255.0
    mask_3  = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    img_f   = img_rgb.astype(np.float32)
    result  = img_f * mask_3 + white * (1.0 - mask_3)
    return np.clip(result, 0, 255).astype(np.uint8)


def get_banana_bbox(mask, padding_pct=0.12):
    """Get bounding box of banana region with padding."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 200:
        return None
    x, y, w, h = cv2.boundingRect(largest)
    H, W = mask.shape
    px   = int(w * padding_pct)
    py   = int(h * padding_pct)
    return (max(0, x-px), max(0, y-py),
            min(W, x+w+px), min(H, y+h+py))


def save_rgb_image(img_rgb, output_path, size=(224, 224)):
    """
    Save image as PNG using PIL (preserves RGB exactly).
    NEVER use cv2.imwrite here — it converts to BGR!
    """
    pil_img = Image.fromarray(img_rgb).resize(size, Image.LANCZOS)
    # Save as PNG to avoid JPEG compression artifacts
    out_path = output_path.replace(".jpg", ".png").replace(".jpeg", ".png")
    pil_img.save(out_path, format="PNG")
    return out_path


# ══════════════════════════════════════════════════════════════
#  MAIN FUNCTION
# ══════════════════════════════════════════════════════════════

def extract_banana_roi(input_path: str,
                       output_path: str,
                       target_size: tuple = (224, 224),
                       debug: bool = False) -> dict:
    """
    Converts any banana image to:
    - White background
    - Banana centered and cropped
    - 224x224 RGB PNG
    - Saved with PIL (RGB preserved — matches CNN training format)
    """
    result = {
        "success":     False,
        "output_path": output_path,
        "method":      "none",
        "message":     ""
    }

    # ── Load as RGB ───────────────────────────────────────────
    try:
        img_rgb  = load_image_rgb(input_path)
        # Resize for faster processing
        img_work = np.array(
            Image.fromarray(img_rgb).resize((512, 512), Image.LANCZOS)
        )
    except Exception as e:
        result["message"] = f"Load failed: {e}"
        return result

    # ── Detect banana ─────────────────────────────────────────
    try:
        color_mask = detect_banana_mask(img_work)
        banana_pct = np.sum(color_mask > 0) / (512 * 512)

        if banana_pct > 0.05:
            # Refine with GrabCut
            try:
                refined_mask = grabcut_refine(img_work, color_mask)
            except Exception:
                refined_mask = color_mask

            # White background (RGB)
            white_img = make_white_background_rgb(img_work, refined_mask)

            # Crop to banana
            bbox = get_banana_bbox(refined_mask)
            if bbox:
                x1, y1, x2, y2 = bbox
                cropped = white_img[y1:y2, x1:x2]

                if cropped.shape[0] > 20 and cropped.shape[1] > 20:
                    # Save as PNG using PIL (RGB preserved!)
                    out_path = save_rgb_image(cropped, output_path, target_size)
                    result.update({
                        "success": True,
                        "method":  "color+grabcut+whitebg",
                        "output_path": out_path,
                        "message": f"Banana detected ({banana_pct*100:.1f}%). "
                                   f"Saved as RGB PNG."
                    })
                    if debug:
                        base = os.path.splitext(out_path)[0]
                        Image.fromarray(refined_mask).save(f"{base}_mask.png")
                        Image.fromarray(white_img).save(f"{base}_white.png")
                        Image.fromarray(cropped).save(f"{base}_cropped.png")
                    return result

        # Banana too small — apply white bg to full image
        white_img = make_white_background_rgb(img_work, color_mask)
        out_path  = save_rgb_image(white_img, output_path, target_size)
        result.update({
            "success": True,
            "method":  "whitebg_only",
            "output_path": out_path,
            "message": f"White BG applied. Coverage: {banana_pct*100:.1f}%"
        })
        return result

    except Exception as e:
        result["message"] = f"Segmentation error: {e}"

    # ── Fallback: resize original with PIL ────────────────────
    try:
        out_path = save_rgb_image(img_rgb, output_path, target_size)
        result.update({
            "success": True,
            "method":  "fallback_resize",
            "output_path": out_path,
            "message": "Used original image resized."
        })
    except Exception as e:
        result["message"] += f" | Fallback failed: {e}"

    return result


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        r = extract_banana_roi(sys.argv[1],
                               sys.argv[1].replace(".", "_seg."),
                               debug=True)
        print(f"Success : {r['success']}")
        print(f"Method  : {r['method']}")
        print(f"Output  : {r['output_path']}")
        print(f"Message : {r['message']}")
    else:
        print("Usage: python segmentation.py <image_path>")
