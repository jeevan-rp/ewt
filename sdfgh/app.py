from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io, requests, base64

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("Efficient_classify.keras")

# Your class names
class_names = ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse',
               'PCB', 'Player', 'Printer', 'Television', 'Washing Machine']

def preprocess_image(img):
    img = img.resize((128, 128))  # must match training size
    arr = np.array(img, dtype=np.float32)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict(img):
    arr = preprocess_image(img)
    preds = model.predict(arr)[0]
    idx = np.argmax(preds)
    top3_idx = preds.argsort()[-3:][::-1]
    return {
        "label": class_names[idx],
        "confidence": float(preds[idx]),
        "top3": [
            {"label": class_names[i], "confidence": float(preds[i])}
            for i in top3_idx
        ]
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def classify():
    if "image" in request.files:  # uploaded file
        img = Image.open(request.files["image"].stream).convert("RGB")
    elif "url" in request.json:  # URL case
        response = requests.get(request.json["url"])
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        return jsonify({"error": "No image provided"}), 400

    result = predict(img)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
