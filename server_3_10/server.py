from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import random
import tensorflow as tf
import pickle
# import keras
from keras.saving import register_keras_serializable
from predict import percentage_np

# keras.config.enable_unsafe_deserialization()

app = Flask(__name__)
CORS(app)

# === Rutas a tus recursos ===
MODEL_PATH = "../model/model_signature_free.keras"
LABEL_PATH = "../model/label_signature.pkl"
IMAGE_DATA_PATH = "../model/train_image_matrices_150_5.pkl"


@register_keras_serializable()
class L1Distance(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        x, y = inputs
        return tf.math.abs(x - y)


# === Cargar modelo ===
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"L1Distance": L1Distance})

# === Cargar etiquetas ===
with open(LABEL_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# === Cargar imágenes matriciales (estructura: [[forge[], genuine[]], ...]) ===
with open(IMAGE_DATA_PATH, "rb") as f:
    image_data = pickle.load(f)


@app.route("/")
def index():
    return send_file("html/write.html")


@app.route("/api/resize_image", methods=["POST"])
def resize_image():
    try:
        data = request.json
        img_b64 = data.get("image")

        # Procesar imagen: decode a PIL
        _, encoded = img_b64.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        img = Image.open(BytesIO(img_bytes)).convert("L")
        original_width, original_height = img.size
        target_width, target_height = (320, 240)

        # Escalado proporcional
        ratio = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        img_resized = img.resize((new_width, new_height), Image.LANCZOS)

        # Canvas blanco
        canvas = Image.new('L', (target_width, target_height), color=255)
        offset_x = (target_width - new_width) // 2
        offset_y = (target_height - new_height) // 2
        canvas.paste(img_resized, (offset_x, offset_y))

        # Convertir a RGB para navegador
        canvas_rgb = Image.merge("RGB", (canvas, canvas, canvas))
        buffered = BytesIO()
        canvas_rgb.save(buffered, format="PNG")
        resized_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({"image": f"data:image/png;base64,{resized_b64}"})

    except Exception as e:
        print(f"Error resizing image: {e}")
        return jsonify({"error": str(e)}), 400


@app.route("/api/get_sample_image", methods=["GET"])
def get_sample_image():
    muestra = random.choice(image_data)
    tipo_idx = random.choice([0, 1])  # 0 = forge, 1 = genuine
    imagen = random.choice(muestra[tipo_idx])

    # Asegurar forma (H, W)
    if imagen.ndim == 3 and imagen.shape[-1] == 1:
        imagen = imagen[:, :, 0]  # quitar dimensión de canal

    # Si es (240, 320) lo pasamos a RGB para mostrarlo en el navegador
    if len(imagen.shape) == 2:
        imagen_rgb = np.stack([imagen] * 3, axis=-1)
    else:
        imagen_rgb = imagen

    img = Image.fromarray(imagen_rgb.astype(np.uint8))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return jsonify({"image": f"data:image/png;base64,{img_b64}"})


def decode_image(img_b64):
    """
    Convierte una imagen base64 a un array NumPy (240, 320, 1) normalizado [0,1]
    """
    _, encoded = img_b64.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(BytesIO(img_bytes)).convert("L").resize((320, 240))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=-1)  # (240, 320, 1)


option = {
    "0": "Identica",
    "1": "Falcificacion"
}


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        img1_b64 = data.get("image1")
        img2_b64 = data.get("image2")

        img1 = decode_image(img1_b64)
        img2 = decode_image(img2_b64)

        a_input = np.expand_dims(img1, axis=0)  # (1, 240, 320, 1)
        b_input = np.expand_dims(img2, axis=0)

        prediction = model.predict([a_input, b_input], verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
        genuine = float(prediction[0][0])
        forge = float(prediction[0][1])

        # Por defecto mostramos el porcentaje de clase 1 (diferencia)
        print(
            f"Resultado predicción: {prediction}\n    Index:{predicted_index}\n    Label: {predicted_label}\n    Option: {option[predicted_label]}\n    genuine: {genuine}\n    forge: {forge}"
        )

        return jsonify({"predict": option[predicted_label], "genuine": genuine, "forge": forge})

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
