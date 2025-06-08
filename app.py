from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model_path = 'plant_disease_model_reduced.h5'
model = keras.models.load_model(model_path)

# Dynamically load class names from the dataset folder
train_dir = 'reduced_dataset/train'
if os.path.exists(train_dir):
    class_names = sorted(os.listdir(train_dir))
else:
    # Fallback to hardcoded class names (optional)
    class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
                   'Potato___healthy', 'Potato___Late_blight', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
                   'Tomato_healthy', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                   'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
                   'Tomato__Tomato_mosaic_virus', 'Tomato__Tomato_YellowLeaf__Curl_Virus']

# Prediction function
def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join('static/images', file.filename)
            file.save(filepath)
            prediction, confidence = predict_image(filepath)
            return render_template('index.html', prediction=prediction, confidence=confidence, image_url=filepath)
    return render_template('index.html')

# Run app
if __name__ == '__main__':
    app.run(debug=True)


