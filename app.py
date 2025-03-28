import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import cv2

# Constants
IMG_SIZE = (224, 224)
MODEL_PATH = "plant_disease_model.h5"
UPLOAD_FOLDER = "static/uploads"

# Load Model
model = load_model(MODEL_PATH)
class_labels = ["Healthy", "Bacterial Spot", "Late Blight", "Powdery Mildew"]  # Update with your classes

# Create Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Grad-CAM for Explainability
def grad_cam(model, img_array, layer_name='Conv_1'):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions)]
    grads = tape.gradient(loss, conv_outputs)
    guided_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = np.sum(guided_grads * conv_outputs, axis=-1)[0]
    cam = cv2.resize(cam, (224, 224))
    return cam

# Flask Routes
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Process Image
            img = Image.open(filepath).resize(IMG_SIZE)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            prediction = model.predict(img_array)
            predicted_class = class_labels[np.argmax(prediction)]
            confidence = np.max(prediction)

            return render_template('index.html', filename=filename, prediction=predicted_class, confidence=confidence)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
