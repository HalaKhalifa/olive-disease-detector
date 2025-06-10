import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score
from tensorflow.keras.models import load_model, Model
from data_loader import get_data_generators
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

MODEL_PATH = "outputs/models/olive_leaf_disease_model.h5"
HISTORY_PATH = "outputs/history/history.pkl"
PLOT_DIR = "outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

model = load_model(MODEL_PATH, compile=False)
DATA_DIR = "data"
_, _, test_gen = get_data_generators(DATA_DIR, image_size=(224, 224), batch_size=32)

# Predict with TTA
def predict_with_tta(model, generator, tta_steps=5):
    preds = []
    for _ in range(tta_steps):
        generator.reset()
        preds.append(model.predict(generator, verbose=0))
    return np.mean(preds, axis=0)

# Predictions
y_true = test_gen.classes
y_pred_probs = predict_with_tta(model, test_gen, tta_steps=5)
y_pred = np.argmax(y_pred_probs, axis=1)
class_labels = list(test_gen.class_indices.keys())

# Reports
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/confusion_matrix.png")
plt.show()

# Training Curves
with open(HISTORY_PATH, "rb") as f:
    history = pickle.load(f)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history["accuracy"], label="Train Acc")
plt.plot(history["val_accuracy"], label="Val Acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history["loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/training_curves.png")
plt.show()

# Filters
def visualize_filters(model, layer_index=1, max_filters=16):
    filters, _ = model.layers[layer_index].get_weights()
    plt.figure(figsize=(10, 6))
    for i in range(min(filters.shape[-1], max_filters)):
        plt.subplot(4, 4, i+1)
        f = filters[:, :, :, i]
        f_mean = np.mean(f, axis=-1)
        plt.imshow(f_mean, cmap='gray')
        plt.axis('off')
    plt.suptitle("First Convolutional Filters")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/filters.png")
    plt.show()

# Feature Maps
sample_image = test_gen[0][0][0]
for layer in model.layers[::-1]:
    if 'conv' in layer.name:
        feature_model = Model(inputs=model.input, outputs=layer.output)
        break
feature_maps = feature_model.predict(np.expand_dims(sample_image, axis=0))

plt.figure(figsize=(15, 15))
for i in range(min(feature_maps.shape[-1], 16)):
    plt.subplot(4, 4, i+1)
    plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
    plt.axis('off')
plt.suptitle("Feature Maps: top_conv")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/feature_maps_top_conv.png")
plt.show()

# Per-Class ROC Curves
y_true_bin = label_binarize(y_true, classes=range(len(class_labels)))

plt.figure(figsize=(10, 8))
for i in range(len(class_labels)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{class_labels[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Per-Class ROC Curves')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/per_class_roc.png")
plt.show()

# Precision-Recall Curves per Class
plt.figure(figsize=(10, 8))
for i in range(len(class_labels)):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_probs[:, i])
    ap = average_precision_score(y_true_bin[:, i], y_pred_probs[:, i])
    plt.plot(recall, precision, label=f'{class_labels[i]} (AP = {ap:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/precision_recall_curves.png")
plt.show()

# Misclassified Samples (Image Grid)
misclassified_indices = np.where(y_true != y_pred)[0]
plt.figure(figsize=(12, 8))

for i, idx in enumerate(misclassified_indices[:12]):
    image = test_gen[idx // test_gen.batch_size][0][idx % test_gen.batch_size]
    image = (image * 255).astype("uint8")
    true_label = class_labels[y_true[idx]]
    pred_label = class_labels[y_pred[idx]]

    plt.subplot(3, 4, i + 1)
    plt.imshow(image)
    plt.title(f'True: {true_label}\nPred: {pred_label}', fontsize=9)
    plt.axis('off')

plt.suptitle("Misclassified Samples")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/misclassified_samples.png")
plt.show()



# Class Distribution of Predictions vs. True Labels
plt.figure(figsize=(8, 5))
true_counts = np.bincount(y_true, minlength=len(class_labels))
pred_counts = np.bincount(y_pred, minlength=len(class_labels))

x = np.arange(len(class_labels))
width = 0.35

plt.bar(x - width/2, true_counts, width, label='True')
plt.bar(x + width/2, pred_counts, width, label='Predicted')
plt.xticks(x, class_labels, rotation=45)
plt.ylabel("Count")
plt.title("Class Distribution: True vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/class_distribution.png")
plt.show()

# Learning Rate Over Epochs
if 'lr' in history:
    plt.figure()
    plt.plot(history["lr"], label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title("Learning Rate Over Epochs")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/learning_rate_curve.png")
    plt.show()

# ROC-AUC (macro)
try:
    y_true_bin = tf.keras.utils.to_categorical(y_true, num_classes=len(class_labels))
    roc_auc = roc_auc_score(y_true_bin, y_pred_probs, average="macro", multi_class="ovo")
    print(f"\n🏅 ROC-AUC (macro): {roc_auc:.4f}")
except Exception as e:
    print(f"ROC-AUC computation failed: {e}")