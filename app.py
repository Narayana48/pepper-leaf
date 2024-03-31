import tensorflow as tf
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory


app = Flask(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
IMAGE_SIZE = 150

# Load the pre-trained VGG model
json_file = open('model_vgg.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
cnn_model = model_from_json(loaded_model_json)
cnn_model.load_weights("model_vgg.h5")

# Preprocess an image
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0  # normalize to [0,1] range
    return image

# Read the image from path and preprocess
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

# Predict & classify image
def classify(model, image_path):

    preprocessed_imgage = load_and_preprocess_image(image_path)
    preprocessed_imgage = tf.reshape(
        preprocessed_imgage, (1, IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    prob = cnn_model.predict(preprocessed_imgage)

    # Define class labels
    class_labels = ["Unhealthy Pepper leaf", "Healthy Pepper leaf", "This is not a pepper leaf image"]

    # Check if the highest probability is for other classes
    if prob[0][0] >= 0.5:
        label = class_labels[0]  # Unhealthy Pepper leaf
        classified_prob = prob[0][0]
    elif prob[0][1] >= 0.5:
        label = class_labels[1]  # Healthy Pepper leaf
        classified_prob = prob[0][1]
    else:
        label = class_labels[2]  # This is not a pepper leaf image
        classified_prob = max(prob[0][0], prob[0][1])  # Use the maximum probability

    return label, classified_prob


# Home page
@app.route("/")
def home():
    return render_template("home.html")

# Handle image classification
@app.route("/classify", methods=["POST", "GET"])
def upload_file():
    if request.method == "GET":
        return render_template("home.html")
    else:
        file = request.files["image"]
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(upload_image_path)
        label, prob = classify(cnn_model, upload_image_path)
        prob = round((prob * 100), 2)
        return render_template("classify.html", image_file_name=file.filename, label=label, prob=prob)

# Serve uploaded images
@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)