import tensorflow as tf
import numpy as np

# Scalar
scalar = tf.constant(1)

vector = tf.constant([1, 2, 3])

matrix = tf.constant([[2, 3], [4, 5]])

three_d_tensor = tf.constant([[[1], [3]], [[5], [3]]])

mnist = tf.keras.datasets.mnist

fashion_mnist = tf.keras.datasets.fashion_mnist

x_train = np.array([[0], [1]])
y_train = np.array([[1], [2]])

x_test = np.array([[3], [8]])
y_test = np.array([[3], [5]])

# Keras machine learning model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(1, 1)), tf.keras.layers.Dense(1)])

model.compile(optimizer="adam", loss="mse")

model.fit([[[1]]], [[[1]]])

model.evaluate([[[2]]], [[[2]]])
