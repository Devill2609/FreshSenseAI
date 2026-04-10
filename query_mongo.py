from pymongo import MongoClient
import json

client = MongoClient("mongodb://localhost:27017/")
db = client["freshsense"]
collection = db["predictions"]

latest = list(collection.find().sort("created_at", -1).limit(3))

for doc in latest:
    print(f"\n--- Prediction: {doc.get('original_name')} ---")
    print(f"Class: {doc.get('cls')} | Confidence: {doc.get('confidence')}")
    print(f"Probs: {json.dumps(doc.get('probs'), indent=2)}")
    print(f"Segmentation: {doc.get('seg_method')}")
