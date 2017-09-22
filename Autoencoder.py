from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Parameters

global_step = tf.Variable(0, trainable=False)

learningRate = 0.007

nbEpoch = 25
nbBatch = 256
displayStep = 1

examplesToShow = 10

# Network Parameter

nbInput = 784
nbHidden_1 = 256
nbHidden_2 = 128

# Graph Input

images = tf.placeholder(tf.float32,[None,784])

# Weights and biases

weights = \
    {

        'w1': tf.Variable(tf.random_normal([nbInput,nbHidden_1])),
        'w2': tf.Variable(tf.random_normal([nbHidden_1,nbHidden_2])),
        'w3': tf.Variable(tf.random_normal([nbHidden_2,nbHidden_1])),
        'w4': tf.Variable(tf.random_normal([nbHidden_1,nbInput]))
    }

biases = \
    {
        'b1': tf.Variable(tf.random_normal([nbHidden_1])),
        'b2': tf.Variable(tf.random_normal([nbHidden_2])),
        'b3': tf.Variable(tf.random_normal([nbHidden_1])),
        'b4': tf.Variable(tf.random_normal([nbInput]))
    }


# Create Model

def autoencoder(x):

    encoded_1 = tf.add( tf.matmul(x,weights['w1']), biases['b1'])
    encoded_1 = tf.nn.sigmoid(encoded_1)

    encoded_2 = tf.add( tf.matmul(encoded_1,weights['w2']), biases['b2'])
    encoded_2 = tf.nn.sigmoid(encoded_2)

    decoded_1 = tf.add( tf.matmul(encoded_2,weights['w3']),biases['b3'] )
    decoded_1 = tf.nn.sigmoid(decoded_1)

    decoded_2 = tf.add(tf.matmul(decoded_1, weights['w4']), biases['b4'])
    decoded_2 = tf.nn.sigmoid(decoded_2)

    return decoded_2

# Pred and groundThruth

pred = autoencoder(images)
y_true = images

# Define Loss and Optimizer

loss = tf.reduce_mean(tf.pow(y_true - pred, 2))
optimizer = tf.train.AdamOptimizer(learningRate,beta1=0.5,beta2=0.555).minimize(loss)

# Initialize Variables

init = tf.global_variables_initializer()

# Sess.run()

"""
The only difference between Session and an InteractiveSession is that InteractiveSession 
    makes itself the default session so that 
    you can call run() or eval() without explicitly calling the session.
"""
sess = tf.InteractiveSession()
sess.run(init)

total_batch = int(mnist.train.num_examples/nbBatch)

# Training;

for epoch in range(nbEpoch):
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(nbBatch)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, loss], feed_dict={images: batch_xs})

    # Display logs per epoch step
    if epoch % displayStep == 0:
        print("Epoch:", '%04d' % (epoch + 1),
              "cost=", "{:.9f}".format(c))

print("Optimization Finished!..")

# Applying encode and decode over test set
encode_decode = sess.run(
    pred, feed_dict={images: mnist.test.images[:examplesToShow]})
# Compare original images with their reconstructions
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(examplesToShow):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
f.show()
plt.draw()

plt.show()