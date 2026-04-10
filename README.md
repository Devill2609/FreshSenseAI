# 🍌 FreshSense AI — Full Stack Project

Flask + MongoDB + HTML/CSS/JS · CS1138 Project

---

## 📁 Project Structure

```
freshsense_full/
├── app.py                  ← Flask backend (API + routes)
├── requirements.txt        ← Python dependencies
├── templates/
│   ├── index.html          ← User-facing website
│   └── admin.html          ← Admin panel
└── uploads/                ← Auto-created, stores uploaded images
```

---

## ⚙️ Setup (3 steps)

### Step 1 — Install MongoDB
- Download: https://www.mongodb.com/try/download/community
- Install & start MongoDB (it runs on port 27017 by default)
- No configuration needed — the app creates the database automatically

### Step 2 — Install Python dependencies
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### Step 3 — Run Flask
```bash
python app.py
```

Open in browser:
- **User App** → http://localhost:5000
- **Admin Panel** → http://localhost:5000/admin

---

## 🗺️ Build Phases

### ✅ Phase 2A — RIGHT NOW (No model needed)
Everything works in demo mode:
- Upload banana image → get simulated prediction
- Result saved to MongoDB automatically
- View history, analytics, export CSV
- Admin panel with all records

### ⏳ Phase 2B — Train the CNN (Kaggle Notebook)
Dataset: https://www.kaggle.com/datasets/shahriar26s/banana-ripeness-classification-dataset
- Classes: Unripe → Ripe → Overripe → Rotten
- ~13K banana images
- Run training on Kaggle (free GPU)
- Download `best_model.keras`

### 🔌 Phase 2C — Connect CNN to Flask (1 change!)
In `app.py`, uncomment this block (lines 54–63):
```python
import numpy as np
import tensorflow as tf
CLASS_NAMES = ["unripe", "ripe", "overripe", "rotten"]
MODEL = tf.keras.models.load_model("saved_model/best_model.keras")
MODEL_LOADED = True

def predict_with_model(img_path):
    img  = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    arr  = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    arr  = np.expand_dims(arr, axis=0)
    prob = MODEL.predict(arr, verbose=0)[0]
    idx  = int(np.argmax(prob))
    return CLASS_NAMES[idx], {CLASS_NAMES[i]: float(prob[i]) for i in range(4)}
```

Then set `MODEL_LOADED = True` and restart Flask. Done! 🎉

---

## 🗄️ MongoDB Collections

**Database:** `freshsense`

**Collection:** `predictions`
```json
{
  "_id": "ObjectId",
  "filename": "abc123.jpg",
  "original_name": "banana.jpg",
  "cls": "ripe",
  "label": "Ripe (Fresh)",
  "confidence": 94.3,
  "probs": {"unripe": 2.1, "ripe": 94.3, "overripe": 2.8, "rotten": 0.8},
  "location": "city",
  "shelf_life": {"room": 2, "fridge": 6, "freezer": 20},
  "status": "✅ Perfect to eat now.",
  "model_used": "demo",
  "created_at": "2024-03-15T10:30:00Z"
}
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict` | Upload image → get prediction (saved to DB) |
| GET | `/api/history` | Paginated prediction history |
| GET | `/api/stats` | Dashboard stats + charts data |
| DELETE | `/api/history/<id>` | Delete a record |
| GET | `/api/export/csv` | Download all records as CSV |

---


