import tensorflow as tf

MODEL_PATH = "saved_model/best_banana_model.keras"

print("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("\n--- MODEL SUMMARY ---")
    model.summary()
    
    print("\n--- LAYER DETAILS ---")
    for layer in model.layers:
        print(f"{layer.name} ({layer.__class__.__name__})")
        
except Exception as e:
    print(f"Failed to load: {e}")
