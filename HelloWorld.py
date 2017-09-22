import tensorflow as tf
import numpy as np


giris = tf.constant("Erol is starting on TensorFlow :)")

sess = tf.Session()

print(sess.run(giris))