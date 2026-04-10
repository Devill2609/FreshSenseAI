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

# 3. Load image
img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
raw_arr = tf.keras.preprocessing.image.img_to_array(img)

# 4. Test all 3 common preprocessing methods
methods = {
    "Divide by 255 (Current App.py)": raw_arr / 255.0,
    "No scaling (0 to 255)": raw_arr,
    "MobileNet scaling (-1 to 1)": (raw_arr / 127.5) - 1.0
}

print("\n--- PREDICTION RESULTS FOR DIFFERENT PREPROCESSING ---")
for name, arr in methods.items():
    arr_expanded = np.expand_dims(arr, axis=0)
    prob = model.predict(arr_expanded, verbose=0)[0]
    idx = int(np.argmax(prob))
    conf = float(prob[idx]) * 100
    
    print(f"\n{name}:")
    print(f" -> Predicted: {CLASS_NAMES[idx]} ({conf:.2f}%)")
    for i, c in enumerate(CLASS_NAMES):
        print(f"    {c}: {prob[i]*100:.2f}%")
