import os
from datetime import datetime
from data_loader import get_data_generators
from model_builder import create_mobilenetv2_model

# 🔧 Settings
DATA_DIR = "/Users/halakhalifa/Desktop/Advanced ML/olive-disease-detector/data"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 3
FINE_TUNE_AT = None  # Set to layer index if fine-tuning later

# 📁 Create output folder if not exists
MODEL_SAVE_DIR = "outputs/models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 📦 Load Data
train_gen, val_gen, test_gen = get_data_generators(
    data_dir=DATA_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# 🧠 Build Model
model = create_mobilenetv2_model(
    input_shape=IMAGE_SIZE + (3,),
    num_classes=NUM_CLASSES,
    fine_tune_at=FINE_TUNE_AT
)

# 🏋️‍♀️ Train
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen
)

# 💾 Save model
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
model.save(os.path.join(MODEL_SAVE_DIR, f"mobilenetv2_{timestamp}.h5"))

print(f"✅ Training complete. Model saved at {MODEL_SAVE_DIR}")
