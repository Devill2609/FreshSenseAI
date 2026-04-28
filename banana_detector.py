"""
banana_detector.py — YOLOv8 Banana Detector
============================================
Detects ALL bananas in an image (basket of fruits etc.)
and returns individual cropped banana images for CNN prediction.

Install:
    pip install ultralytics

YOLOv8 is pre-trained on COCO dataset which includes
'banana' as class 46 — no extra training needed!

Usage:
    from banana_detector import detect_bananas
    results = detect_bananas("basket.jpg", output_dir="uploads/detected")
"""

import os
import cv2
import numpy as np
from PIL import Image

# COCO class index for banana
BANANA_CLASS_ID = 46

# Confidence threshold — only accept high confidence detections
MIN_CONFIDENCE = 0.35

# ── Load YOLO model once at import ────────────────────────────
YOLO_MODEL     = None
YOLO_AVAILABLE = False

def load_yolo():
    """Load YOLOv8 model once. Cached after first call."""
    global YOLO_MODEL, YOLO_AVAILABLE
    if YOLO_MODEL is not None:
        return True
    try:
        from ultralytics import YOLO
        print("[YOLO] Loading YOLOv8n model...")
        # yolov8n = nano (fastest, good enough for detection)
        YOLO_MODEL     = YOLO("yolov8n.pt")   # auto-downloads on first run
        YOLO_AVAILABLE = True
        print("[YOLO] ✅ YOLOv8 loaded!")
        return True
    except ImportError:
        print("[YOLO] ⚠️  ultralytics not installed. Run: pip install ultralytics")
        return False
    except Exception as e:
        print(f"[YOLO] ⚠️  Failed to load: {e}")
        return False


# ══════════════════════════════════════════════════════════════
#  MAIN DETECTION FUNCTION
# ══════════════════════════════════════════════════════════════

def detect_bananas(image_path: str,
                   output_dir: str = "uploads/detected",
                   padding_pct: float = 0.08,
                   target_size: tuple = (224, 224)) -> dict:
    """
    Detects all bananas in an image and returns cropped versions.

    Args:
        image_path  : Path to input image (basket of fruits etc.)
        output_dir  : Directory to save individual banana crops
        padding_pct : Extra padding around each detected banana
        target_size : Size of each crop for CNN input

    Returns:
        dict with keys:
            success          : bool
            method           : str
            banana_count     : int
            bananas          : list of dicts, each containing:
                                 - crop_path  : str (path to cropped image)
                                 - bbox       : [x1,y1,x2,y2]
                                 - confidence : float
                                 - index      : int
            annotated_path   : str (path to image with bounding boxes drawn)
            message          : str
    """
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "success":       False,
        "method":        "none",
        "banana_count":  0,
        "bananas":       [],
        "annotated_path": None,
        "message":       ""
    }

    # ── Load image ────────────────────────────────────────────
    try:
        img_pil = Image.open(image_path).convert("RGB")
        img_rgb = np.array(img_pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        H, W    = img_rgb.shape[:2]
    except Exception as e:
        result["message"] = f"Cannot load image: {e}"
        return result

    # ── Try YOLOv8 first ──────────────────────────────────────
    if load_yolo():
        try:
            yolo_result = YOLO_MODEL(
                image_path,
                conf   = MIN_CONFIDENCE,
                classes= [BANANA_CLASS_ID],   # only detect bananas
                verbose= False
            )[0]

            boxes      = yolo_result.boxes
            banana_boxes = []

            for box in boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                if cls_id == BANANA_CLASS_ID and conf >= MIN_CONFIDENCE:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    banana_boxes.append({
                        "bbox":       [x1, y1, x2, y2],
                        "confidence": round(conf, 4)
                    })

            if banana_boxes:
                # Sort by size (largest first)
                banana_boxes.sort(
                    key=lambda b: (b["bbox"][2]-b["bbox"][0]) *
                                  (b["bbox"][3]-b["bbox"][1]),
                    reverse=True
                )

                # Draw annotated image
                annotated_bgr  = img_bgr.copy()
                banana_results = []

                for i, bdata in enumerate(banana_boxes):
                    x1, y1, x2, y2 = bdata["bbox"]
                    conf            = bdata["confidence"]

                    # Add padding
                    px  = int((x2-x1) * padding_pct)
                    py  = int((y2-y1) * padding_pct)
                    x1p = max(0, x1 - px)
                    y1p = max(0, y1 - py)
                    x2p = min(W, x2 + px)
                    y2p = min(H, y2 + py)

                    # Crop banana (RGB)
                    crop_rgb  = img_rgb[y1p:y2p, x1p:x2p]
                    crop_pil  = Image.fromarray(crop_rgb).resize(
                        target_size, Image.LANCZOS
                    )
                    crop_path = os.path.join(
                        output_dir, f"banana_{i+1}.png"
                    )
                    crop_pil.save(crop_path, format="PNG")

                    # Draw box on annotated image
                    color = (57, 217, 138)   # green
                    cv2.rectangle(annotated_bgr,
                                  (x1, y1), (x2, y2), color, 3)
                    label = f"Banana {i+1} ({conf*100:.0f}%)"
                    cv2.rectangle(annotated_bgr,
                                  (x1, y1-30), (x1+len(label)*11, y1),
                                  color, -1)
                    cv2.putText(annotated_bgr, label,
                                (x1+4, y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (0, 0, 0), 2)

                    banana_results.append({
                        "index":      i + 1,
                        "crop_path":  crop_path,
                        "bbox":       [x1, y1, x2, y2],
                        "confidence": conf,
                    })

                # Save annotated image
                base_name      = os.path.splitext(
                    os.path.basename(image_path))[0]
                annotated_path = os.path.join(
                    output_dir, f"{base_name}_annotated.jpg"
                )
                cv2.imwrite(annotated_path, annotated_bgr)

                result.update({
                    "success":        True,
                    "method":         "yolov8",
                    "banana_count":   len(banana_results),
                    "bananas":        banana_results,
                    "annotated_path": annotated_path,
                    "message":        f"YOLOv8 detected {len(banana_results)} banana(s)."
                })
                return result

            else:
                result["message"] = "YOLOv8 found no bananas. Trying color fallback..."

        except Exception as e:
            result["message"] = f"YOLOv8 error: {e}. Trying color fallback..."
            print(f"[YOLO] Error: {e}")

    # ── Fallback: Color-based banana detection ────────────────
    print("[DETECT] Using color-based fallback detection...")
    try:
        img_work = cv2.resize(img_bgr, (640, 640))
        scale_x  = W / 640
        scale_y  = H / 640

        hsv = cv2.cvtColor(img_work, cv2.COLOR_BGR2HSV)

        # Banana color ranges
        masks = [
            cv2.inRange(hsv, np.array([15, 50, 80]),  np.array([40, 255, 255])),  # yellow
            cv2.inRange(hsv, np.array([25, 30, 30]),  np.array([90, 255, 255])),  # green
            cv2.inRange(hsv, np.array([10, 30, 40]),  np.array([25, 220, 200])),  # overripe
        ]
        combined = masks[0]
        for m in masks[1:]:
            combined = cv2.bitwise_or(combined, m)

        # Morphological cleanup
        kernel   = np.ones((15, 15), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        # Filter by area (at least 2% of image)
        min_area    = 640 * 640 * 0.02
        valid       = [c for c in contours if cv2.contourArea(c) > min_area]

        if not valid:
            # Nothing found — return full image as single banana
            crop_path = os.path.join(output_dir, "banana_1.png")
            img_pil.resize(target_size, Image.LANCZOS).save(crop_path)
            result.update({
                "success":      True,
                "method":       "fallback_full_image",
                "banana_count": 1,
                "bananas":      [{"index":1,"crop_path":crop_path,
                                  "bbox":[0,0,W,H],"confidence":0.5}],
                "message":      "No banana regions detected. Using full image."
            })
            return result

        # Sort by area
        valid.sort(key=cv2.contourArea, reverse=True)
        valid = valid[:5]  # max 5 bananas

        banana_results = []
        annotated_bgr  = img_bgr.copy()

        for i, cnt in enumerate(valid):
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Scale back to original size
            x1 = int(x * scale_x); y1 = int(y * scale_y)
            x2 = int((x+bw) * scale_x); y2 = int((y+bh) * scale_y)

            # Padding
            px  = int((x2-x1) * padding_pct)
            py  = int((y2-y1) * padding_pct)
            x1p = max(0, x1-px); y1p = max(0, y1-py)
            x2p = min(W, x2+px); y2p = min(H, y2+py)

            crop_rgb  = img_rgb[y1p:y2p, x1p:x2p]
            crop_pil  = Image.fromarray(crop_rgb).resize(target_size, Image.LANCZOS)
            crop_path = os.path.join(output_dir, f"banana_{i+1}.png")
            crop_pil.save(crop_path, format="PNG")

            # Draw box
            cv2.rectangle(annotated_bgr, (x1,y1), (x2,y2), (57,217,138), 3)
            cv2.putText(annotated_bgr, f"Banana {i+1}",
                        (x1+4, y1-8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (57,217,138), 2)

            banana_results.append({
                "index":      i+1,
                "crop_path":  crop_path,
                "bbox":       [x1, y1, x2, y2],
                "confidence": 0.6,
            })

        base_name      = os.path.splitext(os.path.basename(image_path))[0]
        annotated_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
        cv2.imwrite(annotated_path, annotated_bgr)

        result.update({
            "success":        True,
            "method":         "color_fallback",
            "banana_count":   len(banana_results),
            "bananas":        banana_results,
            "annotated_path": annotated_path,
            "message":        f"Color detection found {len(banana_results)} banana region(s)."
        })

    except Exception as e:
        result["message"] += f" | Color fallback failed: {e}"

    return result


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json
    if len(sys.argv) > 1:
        r = detect_bananas(sys.argv[1], output_dir="./detected_test")
        print(json.dumps({
            "success":      r["success"],
            "method":       r["method"],
            "banana_count": r["banana_count"],
            "message":      r["message"],
            "crops":        [b["crop_path"] for b in r["bananas"]],
        }, indent=2))
    else:
        print("Usage: python banana_detector.py <image_path>")
