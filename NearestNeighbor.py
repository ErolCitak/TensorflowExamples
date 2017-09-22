import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In this example, we limit mnist data
xTr, yTr = mnist.train.next_batch(5000)
xTe, yTe = mnist.test.next_batch(200)

#Graph Input

xtr = tf.placeholder(tf.float32,[None,784])
xte = tf.placeholder(tf.float32,[784])

# Nearest Neighbor calculation using L1 Distance
# Calculate L1 Distance
distance = tf.reduce_sum(tf.abs(tf.add(xtr,tf.negative(xte))), reduction_indices=1)

#prediction: Get min distance index
pred = tf.arg_min(distance,0)

accuracy = 0

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(len(xTe)):
        nn_index = sess.run(pred,feed_dict={xtr:xTr, xte:xTe[i,:]})
        # Get nearest neighbor class label and compare it to its true label
        print("Test", i, "Prediction:", np.argmax(yTr[nn_index]),"True Class:", np.argmax(yTe[i]))
        # Calculate accuracy
        if np.argmax(yTr[nn_index]) == np.argmax(yTe[i]):
            accuracy += 1. / len(xTe)

print("Done!")
print("Accuracy:", accuracy)
