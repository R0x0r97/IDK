import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

with tf.Session() as sesh:
    y_train = sesh.run(tf.one_hot(y_train, 10))
    y_test = sesh.run(tf.one_hot(y_test, 10))


#hyperparameters - parameters that are set and not learnt
learning_rate = 0.01
epochs = 10
batch_size = 1000
batches = int(x_train.shape[0] / batch_size)

# Y = sigma * (X * W + B)

#inputs
# X is our "flattened / normalized"(from preprocessing) images
# Y is our "one hot" labels

#actual data
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

#trained(learned) variables
W = tf.Variable(0.001 * np.random.rand(784, 10).astype(np.float32))
B = tf.Variable(0.001 * np.random.rand(10).astype(np.float32))

# graph = categorization of X*W+B
pred = tf.nn.softmax(tf.add(tf.matmul(X, W), B))

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

with tf.Session() as sesh:
    sesh.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        for i in range(batches):
            #offset = i * epoch
            x = x_train[i*batch_size: (i+1)*batch_size]
            y = y_train[i*batch_size: (i+1)*batch_size]
            #x = x_train[offset: offset + batch_size]
            #y = y_train[offset: offset + batch_size]
            sesh.run(optimizer, feed_dict = {X: x, Y: y})
            c = sesh.run(cost, feed_dict = {X: x, Y: y})
        if not epoch % 1:
            print(f'epoch:{epoch} cost={c:.5f}')
    
    weigths = sesh.run(W) 
    biases = sesh.run(B)
    np.savetxt("Python\\IDK\\Weitghs.txt", weigths, fmt='%.8f')
    np.savetxt("Python\\IDK\\Biases.txt", biases, fmt='%.8f')

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    accuracy2 = accuracy.eval({X: x_test, Y: y_test})
    print(accuracy2)

    nr_of_tests_to_show = 30
    fig, axes = plt.subplots(1, nr_of_tests_to_show, figsize = (8, 4))
    for img, ax in zip(x_test[:nr_of_tests_to_show], axes):
        guess = np.argmax(sesh.run(pred, feed_dict = {X: [img]}))
        ax.set_title(guess)
        ax.imshow(img.reshape((28,28)))
        ax.axis('off')
    plt.show()
