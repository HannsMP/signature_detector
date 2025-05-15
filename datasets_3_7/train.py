from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import random
import math
import os

# 1. Cargar dataset MNIST
dataset, metadata = tfds.load('mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']


# 2. Normalizar imágenes (de 0–255 a 0–1)
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label


train_dataset = train_dataset.map(normalize_img)
test_dataset = test_dataset.map(normalize_img)

# 3. Convertir a listas para combinaciones
train_data = list(tfds.as_numpy(train_dataset))
test_data = list(tfds.as_numpy(test_dataset))


# 4. Crear pares de imágenes para entrenamiento (pares positivos y negativos)
def make_pairs(data, num_pairs=60000):
    images, labels = zip(*data)

    pairs = []
    pair_labels = []

    for n in range(0, num_pairs, 2):
        idx1 = n
        idx2 = n + 1
        label1 = labels[idx1]
        label2 = labels[idx2]

        # Positivo
        pairs.append([images[idx1], images[idx1]])
        pair_labels.append(1.0)

        # Positivo
        pairs.append([images[idx2], images[idx2]])
        pair_labels.append(1.0)

        if label2 == label1:
            # Negativo (similares pero no idénticos)
            pairs.append([images[idx1], images[idx2]])
            pair_labels.append(0.5)

            pairs.append([images[idx2], images[idx1]])
            pair_labels.append(0.5)
        else:
            # Negativo (totalmente diferentes)
            pairs.append([images[idx1], images[idx2]])
            pair_labels.append(0.0)

            pairs.append([images[idx2], images[idx1]])
            pair_labels.append(0.0)

    return np.array(pairs), np.array(pair_labels)


train_pairs, train_labels = make_pairs(train_data, num_pairs=60000)
test_pairs, test_labels = make_pairs(test_data, num_pairs=10000)

# 5. Crear dataset de TensorFlow
BATCH_SIZE = 32

train_dataset = tf.data.Dataset.from_tensor_slices(((train_pairs[:, 0], train_pairs[:, 1]), train_labels))
train_dataset = train_dataset.shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices(((test_pairs[:, 0], test_pairs[:, 1]), test_labels))
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# 6. Red convolucional base compartida
def build_base_network(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Reshape((28, 28, 1))(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    return tf.keras.Model(inputs, x)


input_shape = (28, 28)
base_network = build_base_network(input_shape)

# 7. Inputs de la red siamesa
input_a = tf.keras.Input(shape=input_shape)
input_b = tf.keras.Input(shape=input_shape)

processed_a = base_network(input_a)
processed_b = base_network(input_b)

# 8. Distancia absoluta entre codificaciones
L1_layer = tf.keras.layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([processed_a, processed_b])

# 9. Capa final
prediction = tf.keras.layers.Dense(1)(L1_distance)

# 10. Modelo completo
siamese_net = tf.keras.Model(inputs=[input_a, input_b], outputs=prediction)

siamese_net.compile(loss='mse', optimizer='adam', metrics=['mae'])

# 11. Entrenamiento
siamese_net.fit(train_dataset, epochs=10, validation_data=test_dataset)

# 12. Guardar el modelo
os.makedirs("modelo_siamese_firmas", exist_ok=True)
siamese_net.save("modelo_siamese_firmas")

print("✅ Entrenamiento completado y modelo guardado.")
