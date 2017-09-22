import tensorflow as tf
import numpy as np
"""
### DEFINING GRAPH ###

b = tf.Variable(tf.zeros(100,))

# -1 ile +1 arasinda W olustur random uniform...
W = tf.Variable(tf.random_uniform((784,100),-1,1))

x = tf.placeholder(tf.float32,(100,784))

h = tf.nn.relu(tf.matmul(x,W) + b)


# We define the LOSS...
prediction = tf.nn.softmax(h)
label = tf.placeholder(tf.float32, [None,10])

cross_entropy = -tf.reduce_sum(label * tf.log(prediction), axis=1)


#time to TRAIN...
# 0.5 is a learning rate....
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


sess = tf.Session()
sess.run(tf.initialize_all_variables())


for i in range(1000):
    # h our graph, {} refers the input value for placeholder.
    batch_x, batch_label =   .next_batch()
    sess.run(train_step, feed_dict={ x: batch_x,label: batch_label})
"""