import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters

learningRate = 0.001
nbIter = 100000
nbBatch = 10
displayStep = 10

# Network Parameters

nInput = 28
nSteps = 28
nHidden = 128
nClasses = 10

# tf Graph Input

x = tf.placeholder(tf.float32,[None,nSteps,nInput])
y = tf.placeholder(tf.float32,[None,nClasses])

# Weights and Biases

weights = \
    {
        'out': tf.Variable(tf.random_normal([nHidden,nClasses]))
    }

biases = \
    {
        'out': tf.Variable(tf.random_normal([nClasses]))
    }

# Create Model

def RNN(x , weights ,biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batchSize, nInput)
    x = tf.unstack(x,nSteps,1)

    # Define LSTM cell
    lstmCell = rnn.BasicLSTMCell(nHidden,forget_bias=1.01)

    # Get LSTM Cell output
    output, states = rnn.static_rnn(lstmCell,x,dtype=tf.float32)

    return tf.matmul(output[-1],weights['out']) + biases['out']

# Pred

pred = RNN(x,weights,biases)

# Define Loss and Optimizer

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize Variables

init = tf.global_variables_initializer()


# Sess.run()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    # Keep Training until reach max iterations
    while step * nbBatch  < nbIter:
        batch_x, batch_y = mnist.train.next_batch(nbBatch)

        batch_x = batch_x.reshape((nbBatch,nSteps,nInput))

        # Run Optimization op (Backprop)
        sess.run(optimizer, feed_dict={x:batch_x,y:batch_y})

        if step % displayStep == 0:
            # Calculate Batch Accuracy
            acc,loss = sess.run([accuracy,cost], feed_dict={x:batch_x,y:batch_y})

            #Calculate Batch Loss
            #loss = sess.run(cost, feed_dict={x:batch_x,y:batch_y})

            print( "Iter " + str(step * nbBatch) + ", Minibatch Loss= " + \
            "{:.6f}".format(loss) + ", Training Accuracy= " + \
            "{:.5f}".format(acc))

        step += 1

    print("Optimization Finished!...")

    # Calculate accuracy for 128 mnist test images

    testLen = 128
    testData = mnist.test.images[:testLen].reshape((-1, nSteps, nInput))
    testLabel = mnist.test.labels[:testLen]

    print("Test Accuracy for 128 Mnist test images: ", sess.run(accuracy,
                                                                feed_dict={x:testData , y:testLabel}))