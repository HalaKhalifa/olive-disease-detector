import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import get_data_generators

# Paths
DATA_DIR = "/Users/halakhalifa/Desktop/Advanced ML/olive-disease-detector/data"
MODEL_PATH = "outputs/models"  # Path to saved models

# Get the latest saved model (you can hardcode if you prefer)
model_files = sorted(os.listdir(MODEL_PATH))
latest_model_path = os.path.join(MODEL_PATH, model_files[-1])

print(f"🧠 Loading model: {latest_model_path}")
model = load_model(latest_model_path)

# Load test data
_, _, test_gen = get_data_generators(
    data_dir=DATA_DIR,
    image_size=(224, 224),
    batch_size=32
)

# Predict
y_probs = model.predict(test_gen)
y_pred = np.argmax(y_probs, axis=1)
y_true = test_gen.classes

# Class labels
class_labels = list(test_gen.class_indices.keys())

# Evaluation report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Visualize confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("outputs/plots/confusion_matrix.png")
plt.show()