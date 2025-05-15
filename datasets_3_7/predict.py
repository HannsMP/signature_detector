import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import random

# 1. Cargar modelo
model = tf.keras.models.load_model("modelo_siamese_firmas")

# 2. Cargar dataset y normalizar
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

dataset, metadata = tfds.load('mnist', as_supervised=True, with_info=True)
test_dataset = dataset['test'].map(normalize_img)
test_data = list(tfds.as_numpy(test_dataset))

# 3. Escoger dos imágenes aleatorias
(img1, lbl1), (img2, lbl2) = random.sample(test_data, 2)

# 4. Preparar combinaciones
pairs = [
    (img1, img1, "img1 vs img1"),
    (img1, img2, "img1 vs img2"),
    (img2, img1, "img2 vs img1"),
    (img2, img2, "img2 vs img2"),
]

# 5. Realizar predicciones
predictions = []
for a, b, label in pairs:
    a_input = np.expand_dims(a, axis=0)  # (1, 28, 28)
    b_input = np.expand_dims(b, axis=0)
    pred = model.predict([a_input, b_input], verbose=0)
    print(pred)
    predictions.append((a, b, pred[0][0], label))

# 6. Visualizar resultados
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for i, (img_a, img_b, pred, label) in enumerate(predictions):
    ax1 = axes[0, i]
    ax2 = axes[1, i]

    ax1.imshow(img_a, cmap='gray')
    ax1.axis('off')

    ax2.imshow(img_b, cmap='gray')
    ax2.axis('off')
    ax2.set_title(f"{label}\nSimilitud: {pred*100:.2f}%")

plt.tight_layout()
plt.show()