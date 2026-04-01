import os
import sys
import numpy as np
import tensorflow as tf
import cv2

# Configuration
MODEL_PATH = "../models/flameguard_msanet.h5"
IMG_SIZE = (128, 128)

def predict_image(image_path):
    """Loads a single image, processes it, and predicts Fire or Non-Fire."""
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}.")
        print("Please ensure you have trained the model or downloaded the weights.")
        return

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}.")
        return

    # Load and preprocess the image
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV loads as BGR, convert to RGB
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0 # Normalize
        img = np.expand_dims(img, axis=0) # Add batch dimension
    except Exception as e:
        print(f"Error processing image: {e}")
        return

    # Load model and predict
    print("Analyzing image...")
    model = tf.keras.models.load_model(MODEL_PATH)
    prediction = model.predict(img, verbose=0)[0][0]
    
    # Interpret results
    confidence = (1 - prediction) * 100 if prediction < 0.5 else prediction * 100
    label = "Fire Detected" if prediction < 0.5 else "No Fire (Normal)"
    
    print("-" * 30)
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)
        
    target_image = sys.argv[1]
    predict_image(target_image)
