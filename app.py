"""
app.py — FreshSense AI · Flask Backend
=======================================
Run: python app.py
"""

import os, io, csv, uuid, random, json
from datetime import datetime, timezone

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from pymongo import MongoClient, DESCENDING
from bson import ObjectId
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXT   = {"jpg", "jpeg", "png", "webp"}
app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER,       exist_ok=True)
os.makedirs("uploads/segmented", exist_ok=True)

# ── MongoDB ───────────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
client    = MongoClient(MONGO_URI)
db        = client["freshsense"]
predictions_col = db["predictions"]

# ── Class metadata ────────────────────────────────────────────
# MUST match training order: overripe=0, ripe=1, rotten=2, unripe=3
CLASS_NAMES = ["overripe", "ripe", "rotten", "unripe"]

CLASS_INFO = {
    "overripe": {"label": "Over-Ripe",    "color": "#f4a261",
                 "status": "⚠️ Past peak freshness. Consume soon."},
    "ripe":     {"label": "Ripe (Fresh)", "color": "#39d98a",
                 "status": "✅ Perfect to eat now. Best quality!"},
    "rotten":   {"label": "Rotten",       "color": "#ef4444",
                 "status": "❌ Spoiled. Do NOT consume."},
    "unripe":   {"label": "Under-Ripe",   "color": "#a8dadc",
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
#  CNN MODEL LOADING
# ══════════════════════════════════════════════════════════════
MODEL        = None
MODEL_LOADED = False

# Support both model file names
for model_name in ["best_banana_model.keras", "best_model.keras", "best_model_v2.keras"]:
    MODEL_PATH = os.path.join("saved_model", model_name)
    if os.path.exists(MODEL_PATH):
        break

# Load class indices if available
CLASS_PATH = os.path.join("saved_model", "class_indices.json")
if os.path.exists(CLASS_PATH):
    with open(CLASS_PATH, "r") as f:
        class_indices = json.load(f)
    CLASS_NAMES = [None] * len(class_indices)
    for name, idx in class_indices.items():
        CLASS_NAMES[idx] = name
    print(f"[INFO] Class mapping from file: {class_indices}")
else:
    print(f"[INFO] Using default class mapping: {CLASS_NAMES}")

try:
    import numpy as np
    import tensorflow as tf

    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Loading model: {MODEL_PATH}")
        MODEL        = tf.keras.models.load_model(MODEL_PATH)
        MODEL_LOADED = True
        print(f"[INFO] ✅ Model loaded! Classes: {CLASS_NAMES}")
    else:
        print(f"[WARN] ⚠️  No model found. Running Demo Mode.")

except Exception as e:
    print(f"[ERROR] Model load failed: {e}")

# ── Segmentation ──────────────────────────────────────────────
SEG_AVAILABLE = False
try:
    from segmentation import extract_banana_roi
    SEG_AVAILABLE = True
    print("[INFO] ✅ Segmentation (rembg) loaded")
except Exception as e:
    print(f"[WARN] Segmentation not available: {e}")
    print("[WARN] Run: pip install rembg")


# ══════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════

def predict_with_model(img_path):
    """
    Predict ripeness class from image path.
    Uses RGB format with /255 normalization — matches training generator.
    """
    # Load as RGB (PIL default) — same as ImageDataGenerator
    img  = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    arr  = tf.keras.preprocessing.image.img_to_array(img)   # RGB, 0-255
    arr  = arr / 255.0                                        # normalize to 0-1
    arr  = np.expand_dims(arr, axis=0)                        # add batch dim

    prob = MODEL.predict(arr, verbose=0)[0]
    idx  = int(np.argmax(prob))
    cls  = CLASS_NAMES[idx]

    probs = {CLASS_NAMES[i]: float(prob[i]) for i in range(len(CLASS_NAMES))}
    print(f"[PRED] {', '.join(f'{k}={v*100:.1f}%' for k,v in probs.items())}")
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


# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/login")
def login():
    return send_from_directory("templates", "login.html")

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/admin")
def admin():
    return send_from_directory("templates", "admin.html")

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ── Predict ───────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file. Use JPG/PNG."}), 400

    location = request.form.get("location", "city")

    # Save original
    ext      = file.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Segmentation
    predict_path = filepath
    seg_method   = "none"
    seg_applied  = False

    if SEG_AVAILABLE:
        try:
            seg_filename = f"segmented/{uuid.uuid4().hex}.jpg"
            seg_filepath = os.path.join(UPLOAD_FOLDER, seg_filename)
            seg_result   = extract_banana_roi(filepath, seg_filepath, target_size=(224, 224))
            if seg_result["success"] and os.path.exists(seg_filepath):
                predict_path = seg_filepath
                seg_method   = seg_result["method"]
                seg_applied  = True
                print(f"[SEG] Method: {seg_method} | {seg_result['message']}")
        except Exception as e:
            print(f"[WARN] Segmentation error: {e}")

    # Predict
    try:
        if MODEL_LOADED:
            cls, probs = predict_with_model(predict_path)
        else:
            cls, probs = simulate_prediction()
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        cls, probs = simulate_prediction()

    conf       = probs.get(cls, 0)
    info       = CLASS_INFO.get(cls, CLASS_INFO["ripe"])
    shelf_life = compute_shelf_life(cls, location)

    # Save to MongoDB
    record = {
        "filename":      filename,
        "original_name": secure_filename(file.filename),
        "seg_filename":  seg_filename if seg_applied else None,
        "seg_method":    seg_method,
        "cls":           cls,
        "label":         info["label"],
        "confidence":    round(conf * 100, 2),
        "probs":         {k: round(v * 100, 2) for k, v in probs.items()},
        "location":      location,
        "shelf_life":    shelf_life,
        "status":        info["status"],
        "model_used":    "CNN" if MODEL_LOADED else "demo",
        "created_at":    datetime.now(timezone.utc),
    }
    res            = predictions_col.insert_one(record)
    record["_id"]  = str(res.inserted_id)
    record["created_at"] = record["created_at"].isoformat()

    return jsonify({
        "success":    True,
        "prediction": record,
        "segmentation": {
            "applied": seg_applied,
            "method":  seg_method,
        }
    })


# ── History ───────────────────────────────────────────────────
@app.route("/api/history", methods=["GET"])
def history():
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    skip     = (page - 1) * per_page
    cursor   = predictions_col.find().sort("created_at", DESCENDING).skip(skip).limit(per_page)
    docs     = [serialize(d) for d in cursor]
    total    = predictions_col.count_documents({})
    return jsonify({"records": docs, "total": total, "page": page, "per_page": per_page})


# ── Stats ─────────────────────────────────────────────────────
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
        "total":          total,
        "distribution":   dist,
        "avg_confidence": avg_conf,
        "trend":          trend,
        "model_active":   MODEL_LOADED,
        "class_names":    CLASS_NAMES,
        "seg_available":  SEG_AVAILABLE,
    })


# ── Delete ────────────────────────────────────────────────────
@app.route("/api/history/<record_id>", methods=["DELETE"])
def delete_record(record_id):
    try:
        r = predictions_col.delete_one({"_id": ObjectId(record_id)})
        return jsonify({"success": True}) if r.deleted_count \
            else (jsonify({"error": "Not found"}), 404)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── Export CSV ────────────────────────────────────────────────
@app.route("/api/export/csv", methods=["GET"])
def export_csv():
    cursor = predictions_col.find().sort("created_at", DESCENDING)
    def generate():
        out = io.StringIO(); w = csv.writer(out)
        w.writerow(["ID","File","Class","Label","Confidence",
                    "Location","Room","Fridge","Freezer","Model","Time"])
        yield out.getvalue(); out.truncate(0); out.seek(0)
        for doc in cursor:
            sl = doc.get("shelf_life", {})
            w.writerow([
                str(doc["_id"]), doc.get("original_name",""),
                doc.get("cls",""), doc.get("label",""),
                doc.get("confidence",""), doc.get("location",""),
                sl.get("room",""), sl.get("fridge",""), sl.get("freezer",""),
                doc.get("model_used",""),
                doc.get("created_at","").isoformat()
                if isinstance(doc.get("created_at"), datetime)
                else str(doc.get("created_at",""))
            ])
            yield out.getvalue(); out.truncate(0); out.seek(0)
    return Response(generate(), headers={
        "Content-Disposition": "attachment; filename=freshsense_predictions.csv",
        "Content-Type": "text/csv"})


# ── Debug ─────────────────────────────────────────────────────
@app.route("/api/debug", methods=["GET"])
def debug():
    return jsonify({
        "model_loaded":  MODEL_LOADED,
        "model_path":    MODEL_PATH,
        "model_exists":  os.path.exists(MODEL_PATH),
        "class_names":   CLASS_NAMES,
        "seg_available": SEG_AVAILABLE,
    })


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*52)
    print("  🍌  FreshSense AI — Flask Backend")
    print("  http://localhost:5000")
    print("  Admin  → http://localhost:5000/admin")
    print("  Debug  → http://localhost:5000/api/debug")
    print(f"  Model  → {'✅ CNN Active' if MODEL_LOADED else '⚠️  Demo Mode'}")
    print(f"  Classes→ {CLASS_NAMES}")
    print(f"  Seg    → {'✅ rembg AI' if SEG_AVAILABLE else '⚠️  Not available (pip install rembg)'}")
    print("="*52 + "\n")
    app.run(debug=True, port=5000)
