from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from predict import classify_image
from PIL import Image
import io

# Load model once
class_names = ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse', 
               'PCB', 'Player', 'Printer', 'Television', 'Washing Machine']
model = tf.keras.models.load_model("Efficient_classify.keras")

app = Flask(__name__)

@app.route("/")
def index():
    # Serves your HTML frontend (put index.html in templates folder)
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    # Convert uploaded file to PIL Image
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Call your existing classify function
    result = classify_image(img, model, class_names)

    # Expected that classify_image returns something like {"label": "Battery", "confidence": 0.92}
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
