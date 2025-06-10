from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'app/static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
MODEL_PATH = 'outputs/models/olive_leaf_disease_model.h5'
model = load_model(MODEL_PATH)

# Class labels (adjust as needed)
class_labels = ['aculus_olearius', 'healthy', 'peacock_spot']

# Helper: preprocess image
def prepare_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'leaf' not in request.files:
            return redirect(request.url)
        file = request.files['leaf']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Predict
            img = prepare_image(filepath)
            preds = model.predict(img)[0]
            pred_idx = np.argmax(preds)
            confidence = preds[pred_idx]
            prediction = class_labels[pred_idx]

            return render_template('result.html',
                                   filename=filename,
                                   prediction=prediction,
                                   confidence=confidence)

    return render_template('index.html')

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)