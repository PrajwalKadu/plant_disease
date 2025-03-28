import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Reduced batch size for faster training
EPOCHS = 1  # Reduced epochs
DATASET_PATH = "dataset/PlantVillage"
MODEL_PATH = "plant_disease_model.h5"

# Data Preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2  # 80% training, 20% validation
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)
val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Load Pretrained Model (MobileNetV2)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model layers to speed up training

# Build Model
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation='relu'),  # Reduced layer size
    keras.layers.Dropout(0.3),
    keras.layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile Model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate for better convergence
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the Model (Reduced Epochs)
model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# Save Model
model.save(MODEL_PATH)
print(f"✅ Model saved successfully at {MODEL_PATH}")
