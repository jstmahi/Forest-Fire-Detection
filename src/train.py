import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from model import build_msa_net

# Configuration paths - update these if your dataset is located elsewhere
TRAIN_DIR = "../dataset/train" 
TEST_DIR = "../dataset/test"   
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 100

def train_model():
    # Data Augmentation and Preprocessing
    train_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_data = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary", shuffle=True
    )
    
    test_data = test_datagen.flow_from_directory(
        TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary", shuffle=False
    )

    # Build and Compile Model
    model = build_msa_net()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks for optimization
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

    # Training Loop
    print("Starting training phase...")
    history = model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=test_data,
        callbacks=[early_stopping, reduce_lr]
    )
    
    # Save the final model
    os.makedirs("../models", exist_ok=True)
    model.save("../models/flameguard_msanet.h5")
    print("Model successfully saved to models/flameguard_msanet.h5")

if __name__ == "__main__":
    train_model()