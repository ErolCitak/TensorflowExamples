import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Import MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters

learningRate = 0.001
nbBatch = 128
nbIteration = 100000
nbDisplayStep = 10

# Network Parameters

inputSize = 784
outputSize = 10

# tf.Graph Input

images = tf.placeholder(tf.float32,[None,inputSize])
labels = tf.placeholder(tf.float32,[None,10])

# Wrappers for Convolutional Neural Network

def conv2d(x, W, b, strides = 1):

    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxPooling(x, k= 2):
    return tf.nn.max_pool(x,ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create Model

def createModel(x,W,b):

    # Reshape Input Data
    x = tf.reshape(x, [-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxPooling(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2= maxPooling(conv2, k=2)


    # Fully Connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    fc1 = tf.contrib.layers.batch_norm(fc1,
                                      center=True, scale=True,
                                      scope='bn')

    # Output
    out = tf.add(tf.matmul(fc1,weights['wout']),biases['bout'])
    out = tf.nn.sigmoid(out)

    return out


# Weights and Biases

weights = \
    {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5,5,1,32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5,5,32,64])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
        # Fully Connected 2 Output
        'wout': tf.Variable(tf.random_normal([1024,10]))
    }

biases = \
    {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'bout': tf.Variable(tf.random_normal([outputSize]))
    }

# Prediction

pred = createModel(images,weights,biases)

# Define loss and optimizer

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * nbBatch < nbIteration:
        batch_x, batch_y = mnist.train.next_batch(nbBatch)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={images: batch_x, labels: batch_y})
        if step % nbDisplayStep == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={images: batch_x,
                                                              labels: batch_y})
            print("Iter " + str(step*nbBatch) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={images: mnist.test.images[:256],
                                      labels: mnist.test.labels[:256]}))
