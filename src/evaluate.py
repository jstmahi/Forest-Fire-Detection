import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

TEST_DIR = "../dataset/test"
MODEL_PATH = "../models/flameguard_msanet.h5"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

def evaluate_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please train the model first.")
        return

    # Load test data
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_data = test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary", shuffle=False
    )

    # Load Model
    print("Loading trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Base Evaluation
    test_loss, test_acc = model.evaluate(test_data)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}\n")

    # Predict classes
    predictions = model.predict(test_data)
    predicted_classes = (predictions > 0.5).astype(int).reshape(-1)
    true_labels = test_data.classes

    # Classification Report
    print("Classification Report:")
    print(classification_report(true_labels, predicted_classes, target_names=["Fire", "Non-Fire"]))

    # Confusion Matrix Visualization
    cm = confusion_matrix(true_labels, predicted_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fire", "Non-Fire"], yticklabels=["Fire", "Non-Fire"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - MSA-Net")
    
    os.makedirs("../assets", exist_ok=True)
    plt.savefig("../assets/confusion_matrix.png")
    print("Confusion matrix saved as assets/confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()
