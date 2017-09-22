import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

rng = np.random


# Load Mnist dataset
# Initialize Params
# Build Model
#  -define loss (cross-entropy), optimizer (sgd)
# Get result

#Parameters

learningRate = 0.01
nbEpoch = 10000
displayStep=25
nbBatch = 50

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Graph Input

images = tf.placeholder(tf.float32,[None,784])
labels = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


# Building Model
# z = input x W + b

z = tf.add(tf.matmul(images,W),b)
z = tf.nn.softmax(z)

# Define cross entropy loss...
loss = tf.reduce_mean(-tf.reduce_sum(labels*tf.log(z), reduction_indices=1))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)


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
    correct_prediction = tf.equal(tf.argmax(z, 1), tf.argmax(y, 1))
    # Calculate accuracy for 3000 examples
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({images: mnist.test.images[:3000], labels: mnist.test.labels[:3000]}))

