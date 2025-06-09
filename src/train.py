import os
import numpy as np
import random
import tensorflow as tf
import pickle
from collections import Counter
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from data_loader import get_data_generators
from model_builder import build_model

# Set seeds
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# Paths
DATA_DIR = "data"
MODEL_DIR = "outputs/models"
HISTORY_DIR = "outputs/history"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# Load data
train_gen, val_gen, _ = get_data_generators(DATA_DIR, image_size=(224, 224), batch_size=32)

# Class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_gen.classes), y=train_gen.classes)
class_weights = dict(enumerate(class_weights))

# Build model
model = build_model(input_shape=(224, 224, 3), num_classes=train_gen.num_classes)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-04)
model.compile(optimizer=optimizer,loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(f"{MODEL_DIR}/olive_leaf_disease_model.h5", save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3),
]

# Train
history = model.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=callbacks
)
print(Counter(train_gen.classes))

# Save training history
with open(f"{HISTORY_DIR}/history.pkl", "wb") as f:
    pickle.dump(history.history, f)