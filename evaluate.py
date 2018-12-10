import tensorflow as tf
import numpy as np

def evaluate_input(input_image_name):
    input_image = np.loadtxt(input_image_name, dtype=float)
    input_image = np.absolute(input_image-255)
    input_image = input_image.reshape(1, 784) / 255
    W = np.loadtxt("Weitghs.txt", dtype=float)
    B = np.loadtxt("Biases.txt", dtype=float)
    X = tf.placeholder(tf.float64, [None, 784])
    
    pred = tf.nn.softmax(tf.add(tf.matmul(X, W), B))

    with tf.Session() as sesh:
        sesh.run(tf.global_variables_initializer())
        guess = np.argmax(sesh.run(pred, feed_dict = {X: input_image}))
    return guess
    
