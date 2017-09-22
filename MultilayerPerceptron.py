"""

A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Trainer: Erol Citak
Project: https://github.com/aymericdamien/TensorFlow-Examples/

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

rng = np.random


# Load Mnist Dataset
# create weights and biases group
# Build model (Dense Layers - 2->4)
# Define Loss and optimizer
# Init variable
# Run Sess

# Parameters

beta = 0.01
learningRate = 0.001
nbEpoch = 15
displayStep=1
nbBatch = 256

# Network Parameters

nHidden_1 = 512
nHidden_2 = 512
nHidden_3 = 512
nInput = 784
nClass = 10

# Load Mnist Dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# Graph Input
images = tf.placeholder(tf.float32,[None,nInput])
labels = tf.placeholder(tf.float32,[None,nClass])

# Store layers weight & bias
weights =  \
    {
        'w1': tf.Variable(tf.random_normal([nInput,nHidden_1])),
        'w2': tf.Variable(tf.random_normal([nHidden_1,nHidden_2])),
        'w3': tf.Variable(tf.random_normal([nHidden_2, nHidden_3])),
        'out': tf.Variable(tf.random_normal([nHidden_3,nClass]))
    }

biases = \
    {
        'b1': tf.Variable(tf.zeros([nHidden_1])),
        'b2': tf.Variable(tf.zeros([nHidden_2])),
        'b3': tf.Variable(tf.zeros([nHidden_3])),
        'out': tf.Variable(tf.zeros([nClass]))
    }

# Create Model
def multiLayerPerceptron(images, weights, biases):

    rH_1 = tf.add(tf.matmul(images,weights['w1']),biases['b1'])
    rH_1 = tf.nn.relu(rH_1)

    rH_2 = tf.add(tf.matmul(rH_1,weights['w2']),biases['b2'])
    rH_2 = tf.nn.relu(rH_2)

    rH_3 = tf.add(tf.matmul(rH_2,weights['w3']),biases['b3'])
    rH_3 = tf.nn.relu(rH_3)

    out = tf.add(tf.matmul(rH_3,weights['out']),biases['out'])

    return out

# Construct model
pred = multiLayerPerceptron(images, weights, biases)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=pred))

# Weight L2 loss...
#regularizers = tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['w2']) + tf.nn.l2_loss(weights['w3'])
#loss = tf.reduce_mean(loss + beta * regularizers)

# Adam Optimizer
optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss)

# Initialize Variables
init = tf.global_variables_initializer()

# Run Session

#Initialize Params...
init = tf.global_variables_initializer()

with tf.Session()  as sess:
    sess.run(init)

    #Training cycle
    for epoch in range(nbEpoch):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/nbBatch)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(nbBatch)
            # Fit training using batch data
            _, c = sess.run([optimizer, loss], feed_dict={images: batch_xs,
                                                          labels: batch_ys})

            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % displayStep == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!"),

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
    # Calculate accuracy for 3000 examples
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval(session=sess, feed_dict={images: mnist.test.images, labels: mnist.test.labels}))