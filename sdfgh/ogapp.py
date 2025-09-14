import gradio as gr
import tensorflow as tf
from predict import classify_image

class_names = ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse', 
               'PCB', 'Player', 'Printer', 'Television', 'Washing Machine']

model = tf.keras.models.load_model("Efficient_classify.keras")

iface = gr.Interface(
    fn=lambda img: classify_image(img, model, class_names),
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="E-Waste Image Classifier",
    description="Upload an e-waste image to identify its category."
)

if __name__ == "__main__":
    iface.launch()
