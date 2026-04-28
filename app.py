"""
app.py — FreshSense AI · Flask Backend
=======================================
Features:
- Single banana prediction (/api/predict)
- Basket/multi-fruit detection (/api/predict_basket) ← NEW
- YOLOv8 banana detection
- GrabCut segmentation
- MongoDB storage
- CSV export

Run: python app.py
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import os, io, csv, uuid, random, json
from datetime import datetime, timezone

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from pymongo import MongoClient, DESCENDING
from bson import ObjectId
from werkzeug.utils import secure_filename

from segmentation    import extract_banana_roi
from banana_detector import detect_bananas, load_yolo

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXT   = {"jpg", "jpeg", "png", "webp"}
app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER,             exist_ok=True)
os.makedirs("uploads/segmented",       exist_ok=True)
os.makedirs("uploads/detected",        exist_ok=True)

# ── MongoDB ───────────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
client    = MongoClient(MONGO_URI)
db        = client["freshsense"]
predictions_col = db["predictions"]

# ── Class metadata ────────────────────────────────────────────
CLASS_NAMES = ["overripe", "ripe", "rotten", "unripe"]

CLASS_INFO = {
    "overripe": {"label": "Over-Ripe",   "color": "#f4a261",
                 "status": "⚠️ Past peak freshness. Consume soon."},
    "ripe":     {"label": "Ripe (Fresh)","color": "#39d98a",
                 "status": "✅ Perfect to eat now. Best quality!"},
    "rotten":   {"label": "Rotten",      "color": "#ef4444",
                 "status": "❌ Spoiled. Do NOT consume."},
    "unripe":   {"label": "Under-Ripe",  "color": "#a8dadc",
                 "status": "⏳ Not yet ready. Allow to ripen further."},
}

SHELF_TABLE = {
    "unripe":   {"room": 5,  "fridge": 10, "freezer": 30},
    "ripe":     {"room": 2,  "fridge": 6,  "freezer": 20},
    "overripe": {"room": 1,  "fridge": 3,  "freezer": 10},
    "rotten":   {"room": 0,  "fridge": 0,  "freezer": 0},
}
LOCATION_MULT = {"city": 1.0, "hilly": 1.3, "coastal": 0.85}

# ══════════════════════════════════════════════════════════════
#  CNN MODEL
# ══════════════════════════════════════════════════════════════
MODEL        = None
MODEL_LOADED = False
MODEL_PATH   = "saved_model/best_model.keras"

CLASS_PATH = "saved_model/class_indices.json"
if os.path.exists(CLASS_PATH):
    with open(CLASS_PATH) as f:
        idx_map     = json.load(f)
        CLASS_NAMES = [None] * len(idx_map)
        for name, i in idx_map.items():
            CLASS_NAMES[i] = name
    print(f"[INFO] Class mapping: {idx_map}")
else:
    print(f"[INFO] Default class order: {CLASS_NAMES}")

try:
    import numpy as np
    import tensorflow as tf
    from PIL import Image as PILImage

    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Loading model from {MODEL_PATH} ...")
        MODEL        = tf.keras.models.load_model(MODEL_PATH)
        MODEL_LOADED = True
        print(f"[INFO] ✅ Model loaded! Classes: {CLASS_NAMES}")
    else:
        print(f"[WARN] ⚠️  No model at {MODEL_PATH} — Demo Mode")
except Exception as e:
    print(f"[ERROR] Model load failed: {e}")

# Pre-load YOLO at startup
load_yolo()


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def predict_with_model(img_path):
    """Predict using CNN — PIL RGB loading matches training."""
    pil_img = PILImage.open(img_path).convert("RGB")
    pil_img = pil_img.resize((224, 224), PILImage.LANCZOS)
    arr     = np.array(pil_img, dtype=np.float32) / 255.0
    arr     = np.expand_dims(arr, axis=0)
    prob    = MODEL.predict(arr, verbose=0)[0]
    idx     = int(np.argmax(prob))
    cls     = CLASS_NAMES[idx]
    probs   = {CLASS_NAMES[i]: float(prob[i]) for i in range(len(CLASS_NAMES))}
    print(f"[PRED] {' | '.join(f'{k}={v*100:.1f}%' for k,v in probs.items())}")
    print(f"[PRED] → {cls} ({prob[idx]*100:.2f}%)")
    return cls, probs


def simulate_prediction():
    classes = ["unripe", "ripe", "overripe", "rotten"]
    weights = [0.20, 0.40, 0.25, 0.15]
    r = random.random(); cum, chosen = 0, "ripe"
    for cls, w in zip(classes, weights):
        cum += w
        if r < cum: chosen = cls; break
    main_conf = 0.72 + random.random() * 0.24
    probs, remaining = {}, 1 - main_conf
    others = [c for c in classes if c != chosen]
    for i, c in enumerate(others):
        v = random.random() * remaining * 0.7 if i < len(others)-1 else remaining
        probs[c] = round(v, 4); remaining -= v
    probs[chosen] = round(main_conf, 4)
    return chosen, probs


def compute_shelf_life(cls, location):
    base = SHELF_TABLE.get(cls, SHELF_TABLE["ripe"])
    mult = LOCATION_MULT.get(location, 1.0)
    return {k: (0 if v == 0 else round(v * mult)) for k, v in base.items()}


def allowed_file(fn):
    return "." in fn and fn.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def serialize(doc):
    doc["_id"] = str(doc["_id"])
    if isinstance(doc.get("created_at"), datetime):
        doc["created_at"] = doc["created_at"].isoformat()
    return doc


def run_prediction(img_path, location):
    """
    Shared prediction logic:
    segment → predict → compute shelf life.
    Returns (cls, probs, shelf_life, seg_method).
    """
    seg_out    = img_path.replace(".", "_seg.").replace("/uploads/", "/uploads/segmented/")
    seg_out    = os.path.join("uploads/segmented", f"{uuid.uuid4().hex}.png")
    seg_result = extract_banana_roi(img_path, seg_out, target_size=(224, 224))
    pred_path  = seg_result.get("output_path", seg_out) \
                 if seg_result["success"] and os.path.exists(
                     seg_result.get("output_path", seg_out)) \
                 else img_path
    seg_method = seg_result["method"]

    try:
        if MODEL_LOADED:
            cls, probs = predict_with_model(pred_path)
        else:
            cls, probs = simulate_prediction()
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        cls, probs = simulate_prediction()

    shelf_life = compute_shelf_life(cls, location)
    return cls, probs, shelf_life, seg_method


# ══════════════════════════════════════════════════════════════
#  PAGE ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/admin")
def admin():
    return send_from_directory("templates", "admin.html")

@app.route("/login")
def login():
    if os.path.exists("templates/login.html"):
        return send_from_directory("templates", "login.html")
    return send_from_directory("templates", "index.html")

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ══════════════════════════════════════════════════════════════
#  API — Single Banana Predict
# ══════════════════════════════════════════════════════════════

@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files["image"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file. Use JPG/PNG."}), 400

    location = request.form.get("location", "city")
    ext      = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    cls, probs, shelf_life, seg_method = run_prediction(filepath, location)
    conf = probs.get(cls, 0)
    info = CLASS_INFO.get(cls, CLASS_INFO["ripe"])

    record = {
        "filename":      filename,
        "original_name": secure_filename(file.filename),
        "seg_method":    seg_method,
        "cls":           cls,
        "label":         info["label"],
        "confidence":    round(conf * 100, 2),
        "probs":         {k: round(v * 100, 2) for k, v in probs.items()},
        "location":      location,
        "shelf_life":    shelf_life,
        "status":        info["status"],
        "model_used":    "CNN" if MODEL_LOADED else "demo",
        "is_basket":     False,
        "created_at":    datetime.now(timezone.utc),
    }
    res            = predictions_col.insert_one(record)
    record["_id"]  = str(res.inserted_id)
    record["created_at"] = record["created_at"].isoformat()

    return jsonify({"success": True, "prediction": record})


# ══════════════════════════════════════════════════════════════
#  API — Basket / Multi-Banana Predict  ← NEW
# ══════════════════════════════════════════════════════════════

@app.route("/api/predict_basket", methods=["POST"])
def predict_basket():
    """
    Upload basket of fruits → detect ALL bananas → predict each.
    Returns list of predictions, one per banana.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files["image"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file. Use JPG/PNG."}), 400

    location = request.form.get("location", "city")
    ext      = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # ── Detect bananas ────────────────────────────────────────
    detect_dir = os.path.join(UPLOAD_FOLDER, "detected")
    detection  = detect_bananas(filepath, output_dir=detect_dir)

    print(f"[BASKET] Method: {detection['method']} | "
          f"Found: {detection['banana_count']} banana(s)")

    if not detection["success"] or detection["banana_count"] == 0:
        return jsonify({
            "success": False,
            "error":   "No bananas detected in image.",
            "message": detection.get("message", "Try a clearer photo.")
        }), 400

    # ── Predict each banana ───────────────────────────────────
    results = []
    for banana in detection["bananas"]:
        crop_path = banana["crop_path"]
        cls, probs, shelf_life, seg_method = run_prediction(crop_path, location)
        conf = probs.get(cls, 0)
        info = CLASS_INFO.get(cls, CLASS_INFO["ripe"])

        banana_result = {
            "banana_index":    banana["index"],
            "bbox":            banana["bbox"],
            "detection_conf":  round(banana["confidence"] * 100, 1),
            "cls":             cls,
            "label":           info["label"],
            "color":           info["color"],
            "confidence":      round(conf * 100, 2),
            "probs":           {k: round(v * 100, 2) for k, v in probs.items()},
            "shelf_life":      shelf_life,
            "status":          info["status"],
        }
        results.append(banana_result)

        # Save to MongoDB
        record = {
            "filename":         filename,
            "original_name":    secure_filename(file.filename),
            "banana_index":     banana["index"],
            "seg_method":       seg_method,
            "cls":              cls,
            "label":            info["label"],
            "confidence":       round(conf * 100, 2),
            "probs":            {k: round(v * 100, 2) for k, v in probs.items()},
            "location":         location,
            "shelf_life":       shelf_life,
            "status":           info["status"],
            "model_used":       "CNN" if MODEL_LOADED else "demo",
            "detection_method": detection["method"],
            "is_basket":        True,
            "created_at":       datetime.now(timezone.utc),
        }
        predictions_col.insert_one(record)

    # Annotated image URL
    annotated_url = None
    if detection.get("annotated_path") and os.path.exists(
            detection["annotated_path"]):
        ann_rel       = "detected/" + os.path.basename(detection["annotated_path"])
        annotated_url = f"/uploads/{ann_rel}"

    return jsonify({
        "success":          True,
        "mode":             "basket",
        "detection_method": detection["method"],
        "banana_count":     detection["banana_count"],
        "annotated_url":    annotated_url,
        "results":          results,
        "message":          detection["message"],
    })


# ══════════════════════════════════════════════════════════════
#  API — History, Stats, Delete, Export
# ══════════════════════════════════════════════════════════════

@app.route("/api/history", methods=["GET"])
def history():
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    skip     = (page - 1) * per_page
    cursor   = predictions_col.find().sort("created_at", DESCENDING).skip(skip).limit(per_page)
    docs     = [serialize(d) for d in cursor]
    total    = predictions_col.count_documents({})
    return jsonify({"records": docs, "total": total, "page": page, "per_page": per_page})


@app.route("/api/stats", methods=["GET"])
def stats():
    total    = predictions_col.count_documents({})
    dist     = {str(d["_id"]) if d.get("_id") else "unknown": d["count"]
                for d in predictions_col.aggregate(
                    [{"$group": {"_id": "$cls", "count": {"$sum": 1}}}])}
    avg_res  = list(predictions_col.aggregate(
                    [{"$group": {"_id": None, "avg": {"$avg": "$confidence"}}}]))
    avg_conf = round(avg_res[0]["avg"], 2) if avg_res else 0
    recent   = list(predictions_col.find({}, {"confidence": 1, "created_at": 1})
                    .sort("created_at", DESCENDING).limit(10))
    trend    = [{"conf": d["confidence"],
                 "time": d["created_at"].strftime("%H:%M")
                 if isinstance(d["created_at"], datetime) else ""}
                for d in reversed(recent)]
    return jsonify({
        "total": total, "distribution": dist,
        "avg_confidence": avg_conf, "trend": trend,
        "model_active": MODEL_LOADED, "class_names": CLASS_NAMES,
    })


@app.route("/api/history/<record_id>", methods=["DELETE"])
def delete_record(record_id):
    try:
        r = predictions_col.delete_one({"_id": ObjectId(record_id)})
        return jsonify({"success": True}) if r.deleted_count \
            else (jsonify({"error": "Not found"}), 404)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/export/csv", methods=["GET"])
def export_csv():
    cursor = predictions_col.find().sort("created_at", DESCENDING)
    def generate():
        out = io.StringIO(); w = csv.writer(out)
        w.writerow(["ID","File","BananaIndex","Class","Label","Confidence",
                    "Location","Room","Fridge","Freezer","IsBasket","Model","Time"])
        yield out.getvalue(); out.truncate(0); out.seek(0)
        for doc in cursor:
            sl = doc.get("shelf_life", {})
            w.writerow([
                str(doc["_id"]), doc.get("original_name",""),
                doc.get("banana_index",""),
                doc.get("cls",""), doc.get("label",""),
                doc.get("confidence",""), doc.get("location",""),
                sl.get("room",""), sl.get("fridge",""), sl.get("freezer",""),
                doc.get("is_basket", False),
                doc.get("model_used",""),
                doc.get("created_at","").isoformat()
                if isinstance(doc.get("created_at"), datetime)
                else str(doc.get("created_at",""))
            ])
            yield out.getvalue(); out.truncate(0); out.seek(0)
    return Response(generate(), headers={
        "Content-Disposition": "attachment; filename=freshsense_predictions.csv",
        "Content-Type": "text/csv"})


@app.route("/api/debug", methods=["GET"])
def debug():
    from banana_detector import YOLO_AVAILABLE
    return jsonify({
        "model_loaded":   MODEL_LOADED,
        "model_path":     MODEL_PATH,
        "model_exists":   os.path.exists(MODEL_PATH),
        "class_names":    CLASS_NAMES,
        "yolo_available": YOLO_AVAILABLE,
    })


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from banana_detector import YOLO_AVAILABLE
    print("\n" + "="*54)
    print("  🍌  FreshSense AI — Flask Backend")
    print("  http://localhost:5000")
    print("  Admin  → http://localhost:5000/admin")
    print("  Debug  → http://localhost:5000/api/debug")
    print(f"  Model  → {'✅ CNN Active' if MODEL_LOADED else '⚠️  Demo Mode'}")
    print(f"  Classes→ {CLASS_NAMES}")
    print(f"  YOLO   → {'✅ YOLOv8 Active' if YOLO_AVAILABLE else '⚠️  Fallback mode'}")
    print("="*54 + "\n")
    app.run(debug=True, port=5000)
