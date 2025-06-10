# 🇵🇸 Olive Leaf Disease Detector 🌿

A deep learning-based application to detect and classify olive leaf diseases using MobileNetV2 — built to support Palestinian agriculture and empower local farmers with accessible AI tools.

<img src="app/static/palestine-olive-logo.png" alt="Logo" width="100"/>

## ✨ Project Highlights

- ✅ Trained MobileNetV2 with custom head on 3 classes: healthy, peacock spot, aculus olearius
- 🎯 Achieved 91% test accuracy and 0.98 macro ROC-AUC
- 🌐 Flask web app for uploading olive leaf images and getting predictions
- 🌿 Localized interface

---

## 📂 Folder Structure
```plaintext

olive-disease-detector/
├── app/
│   ├── static/
│   │   ├── styles.css
│   │   ├── uploads/
│   │   └── palestine-olive-logo.png
│   ├── templates/
│   │   ├── index.html
│   │   └── result.html
│   └── app.py
│
├── data/
│   ├── train/
│   │   ├── aculus_olearius/
│   │   ├── healthy/
│   │   └── peacock_spot/
│   └── test/
│       ├── aculus_olearius/
│       ├── healthy/
│       └── peacock_spot/
│
├── notebooks/
│   ├── data-exploration.ipynb
│   └── model-training.ipynb
│
├── outputs/
│   ├── history/
│   │   └── history.pkl
│   ├── models/
│   │   └── olive_leaf_disease_model.h5
│   └── plots/
│       ├── class_distribution.png
│       ├── confusion_matrix.png
│       ├── feature_maps_top_conv.png
│       ├── misclassified_samples.png
│       ├── per_class_roc.png
│       ├── precision_recall_curves.png
│       └── training_curves.png
│
├── src/
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── model_builder.py
│   ├── train.py
│   └── __init__.py
│
├── README.md
└── requirements.txt
```
## 🧠 Model Summary

- Backbone: MobileNetV2 (pretrained on ImageNet)
- Custom layers: GlobalAveragePooling + LeakyReLU + Dropout + BatchNorm
- Loss: Categorical Crossentropy
- Optimizer: Adam (1e-4)
- Data Augmentation: heavy rotation, zoom, flip, etc.
- Class weighting and partial under-sampling (peacock spot)

## 📊 Performance

| Metric         | Value   |
|----------------|---------|
| Test Accuracy  | ~91%    |
| ROC-AUC (macro)| 0.9841  |
| Precision      | > 0.87  |
| Recall         | > 0.90  |

🔍 Visuals are saved in /outputs/plots.

## 🚀 Run the Web App

First, clone the repository:

```bash
git clone https://github.com/HalaKhalifa/olive-disease-detector.git
cd olive-disease-detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:
```bash
cd app
python app.py
```

Then open http://localhost:5000 in your browser.

<img src="Analyze.png" alt="Analyze" width="100"/>


Upload an olive leaf image to get real-time predictions like this:

<img src="Prediction Result.png" alt="predictions" width="100"/>

---

## 📌 TODO / Future Work

- [ ] 🧠 Improve model generalization by incorporating more diverse and balanced datasets.
- [ ] 🪴 Add support for more olive diseases as new labeled data becomes available.
- [ ] 📱 Develop a mobile app version for on-field usage.
- [ ] 📚 Provide agricultural guides and actionable recommendations for managing each predicted disease.
- [ ] 🇵🇸 Include localized support and Arabic translations to serve farmers in Palestine more directly.
---
## 🫱🏻‍🫲🏽 Collaboration

We believe in collective innovation — and your ideas, skills, or feedback can make a real difference! Whether you're a student, developer, agronomist, designer, or activist, your contribution is welcome.

### 🛠️ How to Contribute

Want to contribute code, ideas, or improvements? Here's how you can get started:

1. Fork this repository and clone it locally.
2. Create a new branch for your feature:
   ```bash
   git checkout -b feature-name
   ```
3. Make your changes, then commit:
   ```bash
   git commit -m "Describe your contribution"
   ```
4. Push and open a Pull Request:
   ```bash
   git push origin feature-name
   ```

Ways you can contribute:

- 🔧 Improve model architecture or training pipeline
- 🧪 Add testing or evaluation scripts
- 🖼️ Expand the dataset (new leaf samples, balanced classes)
- 🌍 Translate the app (e.g., Arabic, French)
- 🎨 Enhance the UI/UX of the web app (HTML/CSS)
- 🩺 Add treatment and prevention suggestions for detected diseases
- 📱 Help build a mobile-friendly or offline-compatible version

### 🌿 Ways to Help (Non-Code)

- Share the tool with Palestinian farmers or agritech communities
- Contribute real images of olive leaves from your region
- Recommend localized solutions or agricultural guides
- Spread awareness and spark discussion about AI in agriculture for Palestine

Together, we can empower Palestinian agriculture with AI and bring hope through technology.
Let’s grow something meaningful — together.

🕊️ For Gaza and Palestine 🌿🇵🇸