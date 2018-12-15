import tensorflow as tf
import numpy as np

def evaluate_input(input_image_name):
    input_image = np.loadtxt(input_image_name, dtype=float)
    input_image = np.absolute(input_image-255)
    input_image = input_image.reshape(1, 784) / 255
    W = np.loadtxt("learning\\Weights.txt", dtype=float)
    B = np.loadtxt("learning\\Biases.txt", dtype=float)
    X = tf.placeholder(tf.float64, [None, 784])
    
    pred = tf.nn.softmax(tf.add(tf.matmul(X, W), B))

    with tf.Session() as sesh:
        sesh.run(tf.global_variables_initializer())
        guess = np.argmax(sesh.run(pred, feed_dict = {X: input_image}))
    return guess
    
def evaluate_inputs(input_image_names, nr_of_inputs):
    input_images = np.empty(nr_of_inputs, 28, 28, dtype=float)
    for i in nr_of_inputs:
        input_images[i] = np.loadtxt(input_image_names[i], dtype=float)
        input_images[i] = np.absolute(input_images[i]-255)
        input_images[i] = input_images[i].reshape(nr_of_inputs, 784) / 255

    W = np.loadtxt("Weights.txt", dtype=float)
    B = np.loadtxt("Biases.txt", dtype=float)
    X = tf.placeholder(tf.float64, [None, 784])
    
    pred = tf.nn.softmax(tf.add(tf.matmul(X, W), B))

    with tf.Session() as sesh:
        sesh.run(tf.global_variables_initializer())
        guesses = np.argmax(sesh.run(pred, feed_dict = {X: input_images}))
    return guesses