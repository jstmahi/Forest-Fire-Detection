train_dir = "/content/drive/MyDrive/Forest Fire Detection/Dataset/train"
test_dir = "/content/drive/MyDrive/Forest Fire Detection/Dataset/test"

img_size = (128, 128)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    shuffle=True
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False
)

print("Class Indices:", train_data.class_indices)



from tensorflow.keras.layers import (Input, Conv2D, Dense, Flatten, Dropout,GlobalAveragePooling2D, BatchNormalization,Activation, Multiply, MaxPooling2D, Reshape)

def se_block(input_tensor, ratio=16):
    channels = input_tensor.shape[-1]
    squeeze = GlobalAveragePooling2D()(input_tensor)
    excitation = Dense(channels // ratio, activation="relu")(squeeze)
    excitation = Dense(channels, activation="sigmoid")(excitation)
    excitation = Reshape((1, 1, channels))(excitation)
    return Multiply()([input_tensor, excitation])

def multi_scale_conv(input_tensor, filters):
    conv3 = Conv2D(filters, (3, 3), padding="same", activation="relu")(input_tensor)
    conv5 = Conv2D(filters, (5, 5), padding="same", activation="relu")(input_tensor)
    conv7 = Conv2D(filters, (7, 7), padding="same", activation="relu")(input_tensor)

    output = tf.keras.layers.Add()([conv3, conv5, conv7])
    output = BatchNormalization()(output)
    return output

def msa_net(input_shape=(128, 128, 3)):
    inputs = Input(shape=input_shape)
    x = multi_scale_conv(inputs, 64)
    x = se_block(x)
    x = multi_scale_conv(x, 128)
    x = se_block(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    return model

model = msa_net()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()



early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

history = model.fit(
    train_data,
    epochs=100,
    validation_data=test_data,
    callbacks=[early_stopping, reduce_lr]
)


test_loss, test_acc = loaded_model.evaluate(test_data)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")


import seaborn as sns

cm = confusion_matrix(true_labels, predicted_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fire", "Non-Fire"], yticklabels=["Fire", "Non-Fire"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(12, 6))
for i, img_name in enumerate(fire_images):
    img_path = os.path.join(test_image_dir, "fire", img_name)
    img_array, img = load_and_preprocess_image(img_path)

    prediction = loaded_model.predict(img_array)[0][0]
    predicted_label = "Non-Fire" if prediction > 0.5 else "Fire"

    plt.subplot(1, 5, i + 1)
    plt.imshow(img)
    plt.title(f"True: Fire\nPred: {predicted_label}",
              color="green" if predicted_label == "Fire" else "red")
    plt.axis("off")

plt.suptitle("Fire Images - True vs Predicted", fontsize=14, color="darkred")
plt.tight_layout()
plt.show()

