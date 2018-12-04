import tensorflow as tf
import numpy as np

def evaluate_input(input_image):
    input_image = input_image.reshape(1, 784) / 255
    W = np.fromfile("Python\\IDK\\Weitghs.txt", dtype=float, count=-1, sep="\n")
    B = np.fromfile("Python\\IDK\\Biases.txt", dtype=float, count=-1, sep="\n")
    X = tf.placeholder(tf.float64, [None, 784])

    pred = tf.nn.softmax(tf.add(tf.matmul(X, W), B))

    with tf.Session() as sesh:
        sesh.run(tf.global_variables_initializer())
        guess = np.argmax(sesh.run(pred, feed_dict = {X: input_image}))
    return guess

