import os
import numpy as np
import tensorflow as tf
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["freshsense"]
docs = list(db["predictions"].find().sort("created_at", -1).limit(2))

CLASS_NAMES = ["overripe", "ripe", "rotten", "unripe"]
model = tf.keras.models.load_model("saved_model/best_banana_model.keras")

for doc in docs: # Should be brown banana then yellow banana
    filepath = os.path.join("uploads", "segmented", doc["seg_filename"].split("/")[-1]) if doc["seg_filename"] else os.path.join("uploads", doc["filename"])
    print(f"\n=============================================")
    print(f"Testing on image: {doc['original_name']} ({filepath})")
    
    img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
    raw_arr_rgb = tf.keras.preprocessing.image.img_to_array(img)
    raw_arr_bgr = raw_arr_rgb[..., ::-1]
    
    methods = {
        "RGB (divided by 255)": raw_arr_rgb / 255.0,
        "BGR (divided by 255)": raw_arr_bgr / 255.0
    }
    
    for name, arr in methods.items():
        arr_expanded = np.expand_dims(arr, axis=0)
        prob = model.predict(arr_expanded, verbose=0)[0]
        
        print(f"\n--- {name} ---")
        # Print raw indices and probabilities
        for i, p in enumerate(prob):
            print(f"Index {i} ({CLASS_NAMES[i]}): {p*100:.2f}%")
