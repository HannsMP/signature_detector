from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import random
import tensorflow as tf
from tensorflow.keras.datasets import mnist

app = Flask(__name__)
CORS(app)

# Cargar MNIST
(x_train, _), (_, _) = mnist.load_data()

model = tf.keras.models.load_model("modelo_siamese_firmas")


@app.route("/")
def index():
    return send_file("html/write.html")


@app.route("/api/get_mnist", methods=["GET"])
def get_mnist_image():
    idx = random.randint(0, len(x_train) - 1)
    img = Image.fromarray(x_train[idx]).convert("L").resize((28, 28))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return jsonify({"image": f"data:image/png;base64,{img_b64}"})


def decode_image(img_b64):
    _, encoded = img_b64.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(BytesIO(img_bytes)).convert("L").resize((28, 28))
    return np.array(img, dtype=np.float32) / 255.0


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        img1_b64 = data.get("image1")
        img2_b64 = data.get("image2")

        img1 = decode_image(img1_b64)
        img2 = decode_image(img2_b64)

        # Similitud inversa al MSE
        a_input = np.expand_dims(img1, axis=0)  # (1, 28, 28)
        b_input = np.expand_dims(img2, axis=0)
        pred = model.predict([a_input, b_input], verbose=0)

        res = float(np.asscalar(pred[0]))
        print(res)

        return jsonify({"percentage": res})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
