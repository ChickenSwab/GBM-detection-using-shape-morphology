import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# --------- Config ----------
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 15

# Your Kaggle root
root_dir = r"D:\GBM_code\archive\Training"
# ---------------------------

# Data generators: 80% train / 20% val split from same root
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,      # split inside this directory
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)

train_gen = datagen.flow_from_directory(
    root_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    class_mode="binary",
    batch_size=BATCH_SIZE,
    subset="training",
    shuffle=True,
)

val_gen = datagen.flow_from_directory(
    root_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    class_mode="binary",
    batch_size=BATCH_SIZE,
    subset="validation",
    shuffle=False,
)

print("Class indices:", train_gen.class_indices)

# --------- CNN model ----------
model = models.Sequential(
    [
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),  # binary: tumor vs notumor
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# --------- Training ----------
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
)

# --------- Save model ----------
model.save(r"D:\GBM_code\cnn_brain_tumor.keras")

print("Model saved to D:\\GBM_code\\cnn_brain_tumor.h5")
