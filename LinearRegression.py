import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

#Parameters

learningRate = 0.01
nbEpoch = 10000
displayStep=25


train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])

nbSamples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(rng.randn(),name="weights")
b = tf.Variable(rng.randn(),name="bias")

# Construct a linear model
z = tf.add(tf.multiply(X,W),b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(tf.add(z,tf.negative(Y)),2)) / (2* nbSamples)

# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

#Launch the graph
with tf.Session() as sess:
    sess.run(init)

    #Fit all training data
    for epoch in range(nbEpoch):
        for(x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x, Y:y})

        #Display logs per epoch step
        if (epoch+1) % displayStep == 0:
            c = sess.run(cost,feed_dict={X:train_X,Y:train_Y})
            print("Epoch:",(epoch+1), "Cost:", cost, c ,"W: ",sess.run(W),"b: ",sess.run(b))

    print("Optimization Finished!..")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b))

    #Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
