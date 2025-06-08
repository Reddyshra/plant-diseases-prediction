import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model_path = 'plant_disease_model_reduced.h5'
model = keras.models.load_model(model_path)

# Use the same class names as in app.py
class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
               'Potato___healthy', 'Potato___Late_blight', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
               'Tomato_healthy', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
               'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
               'Tomato__Tomato_mosaic_virus', 'Tomato__Tomato_YellowLeaf__Curl_Virus']

def predict_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    print(f"Prediction: {predicted_class} ({confidence:.2f}%)")

if __name__ == "__main__":
    test_image_path = input("Enter the path of the test image: ")
    if os.path.exists(test_image_path):
        predict_image(test_image_path)
    else:
        print("Image not found. Please check the path.")


