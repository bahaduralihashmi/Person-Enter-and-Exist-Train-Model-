import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
import numpy as np

# ========================
# CONFIG
# ========================
IMG_SIZE = (96, 96)      # resize images
BATCH_SIZE = 16
EPOCHS = 10

TRAIN_DIR = pathlib.Path("C:/Users/bahad/OneDrive/Desktop/ultralytics/bus_enter_exist/train")
VAL_DIR   = pathlib.Path("C:/Users/bahad/OneDrive/Desktop/ultralytics/bus_enter_exist/val")

# ========================
# DATASET
# ========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="binary",
    color_mode="grayscale",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Normalize to [-1,1]
train_ds = train_ds.map(lambda x, y: (x/127.5 - 1.0, y))
val_ds   = val_ds.map(lambda x, y: (x/127.5 - 1.0, y))

# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1)
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))

# Prefetch for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ========================
# MODEL: Tiny CNN
# ========================
model = keras.Sequential([
    layers.Input(shape=(96, 96, 1)),
    layers.Conv2D(8, (3,3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(16, (3,3), activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3,3), activation="relu"),
    layers.Flatten(),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")  # binary classification
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ========================
# TRAINING
# ========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ========================
# SAVE MODEL
# ========================
model.save("person_tinycnn.h5")
print("✅ Model saved as person_tinycnn.h5")

# ========================
# CONVERT TO QUANTIZED TFLITE (INT8)
# ========================
# Representative dataset for INT8 quantization
def representative_dataset():
    for images, _ in train_ds.take(100):
        # Must be float32
        yield [tf.cast(images, tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("person_tinycnn_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TinyCNN INT8 quantized model saved as person_tinycnn_int8.tflite")
