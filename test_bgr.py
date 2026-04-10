import os
import json
import numpy as np
import tensorflow as tf
from pymongo import MongoClient

# 1. Get the last uploaded image from MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["freshsense"]
doc = db["predictions"].find().sort("created_at", -1).limit(1)[0]
filepath = os.path.join("uploads", "segmented", doc["seg_filename"].split("/")[-1]) if doc["seg_filename"] else os.path.join("uploads", doc["filename"])

print(f"Testing on image: {filepath}")

# 2. Load model
CLASS_NAMES = ["overripe", "ripe", "rotten", "unripe"]
model = tf.keras.models.load_model("saved_model/best_banana_model.keras")

# 3. Load image (RGB)
img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
raw_arr_rgb = tf.keras.preprocessing.image.img_to_array(img)

# Convert to BGR
raw_arr_bgr = raw_arr_rgb[..., ::-1]

# 4. Test BGR scaling
methods = {
    "RGB - Current App.py (Divide by 255)": raw_arr_rgb / 255.0,
    "BGR - OpenCV format (Divide by 255)": raw_arr_bgr / 255.0
}

print("\n--- PREDICTION RESULTS ---")
for name, arr in methods.items():
    arr_expanded = np.expand_dims(arr, axis=0)
    prob = model.predict(arr_expanded, verbose=0)[0]
    idx = int(np.argmax(prob))
    conf = float(prob[idx]) * 100
    
    print(f"\n{name}:")
    print(f" -> Predicted: {CLASS_NAMES[idx]} ({conf:.2f}%)")
    for i, c in enumerate(CLASS_NAMES):
        print(f"    {c}: {prob[i]*100:.2f}%")
