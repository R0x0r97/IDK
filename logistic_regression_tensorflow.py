'''
This module's code was mostly took from: 
https://github.com/markjay4k/Tensorflow-Basics-Series/blob/master/part%202%20-%20Logistic%20Regression.ipynb

'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

# Import MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#example of input data
'''
fig, axes = plt.subplots(1, 4, figsize=(7, 3))
for img, label, ax in zip(x_train[:4], y_train[:4], axes):
    ax.set_title(label)
    ax.imshow(img)
    ax.axis('off')
plt.show();
'''
#rand
# data shape
'''
print(f'train images: {x_train.shape}')
print(f'train labels: {y_train.shape}')
print(f' test images: {x_test.shape}')
print(f' test labels: {y_test.shape}')
'''

#preprocessing
x_train = x_train.reshape(60000, 784) / 255
x_test = x_test.reshape(10000, 784) / 255

with tf.Session() as sess:
    y_train = sess.run(tf.one_hot(y_train, 10))
    y_test = sess.run(tf.one_hot(y_test, 10))


#hyperparameters - parameters that are set and not learnt
learning_rate = 0.13
epochs = 1000
batch_size = 600
batches = int(x_train.shape[0] / batch_size)

# Y = sigma * (X * W + B)

#inputs
# X is our "flattened / normalized"(from preprocessing) images
# Y is our "one hot" labels

#declare where the data will flow in
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
#tf.placeholder() -> must be fed with data during tf.Session()'s

#trained(learned) variables
W = tf.Variable(0.001 * np.random.rand(784, 10).astype(np.float32))
B = tf.Variable(0.001 * np.random.rand(10).astype(np.float32))
#tf.Variable() -> can be used in any tf.Session()


# graph = categorization of X*W+B
pred = tf.nn.softmax(tf.add(tf.matmul(X, W), B))
#This function performs the equivalent of:
#softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
#The softmax "squishes" the inputs so that sum(input) = 1: it's a way of normalizing.

# C = sum (- Y * ln(pred))
# cost function
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(pred), axis = 1))

#demonstration of the cost function
'''
#cost function plotted
x = np.linspace(1/100, 1, 100)
plt.plot(x, np.log(x))
plt.show()
'''

#example data
'''
a = np.log([0.04, 0.13, 0.96, 0.12]
           [0.01, 0.93, 0.06, 0.07])    #pred
b = np.array([[0,    0,    1,    0]
              [1,    0,    0,    0]])   #labels

r_sum = np.sum(-a*b, axis = 1)
r_mean = np.mean(r_sum)
print(r_sum)
print(r_mean)
'''

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
start = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #actual training
    for epoch in range(epochs):
        for i in range(batches):
            x = x_train[i*batch_size: (i+1)*batch_size]
            y = y_train[i*batch_size: (i+1)*batch_size]
            sess.run(optimizer, feed_dict = {X: x, Y: y})
            c = sess.run(cost, feed_dict = {X: x, Y: y})
        if not epoch % 1:
            print(f'epoch:{epoch} cost={c:.10f}')
    end = time.time()
    print(end - start)    
    #saving learnt data
    weigths = sess.run(W) 
    biases = sess.run(B)
    np.savetxt("Weitghs.txt", weigths, fmt='%.8f')
    np.savetxt("Biases.txt", biases, fmt='%.8f')

    #calculate accurary
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    accuracy2 = accuracy.eval({X: x_test, Y: y_test})
    print(accuracy2)

    #viewing few tests
    nr_of_tests_to_show = 5
    fig, axes = plt.subplots(1, nr_of_tests_to_show, figsize = (8, 4))
    for img, ax in zip(x_test[:nr_of_tests_to_show], axes):
        guess = np.argmax(sess.run(pred, feed_dict = {X: [img]}))
        ax.set_title(guess)
        ax.imshow(img.reshape((28,28)))
        ax.axis('off')
    plt.show()
