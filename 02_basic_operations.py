import tensorflow as tf

# Basic constant operations
# The value returned by the constructor represents the output
# of the Constant op.

a = tf.constant(2)
b = tf.constant(4)

# Launch the default graph.
with tf.Session() as sess:
    print("a: ",sess.run(a))
    print("b: ", sess.run(b))
    print("a+b: ", sess.run(a+b))
    print("a*b: ",sess.run(a*b))


########################################################################################################################

# Basic Operations with variable as graph input
# The value returned by the constructor represents the output
# of the Variable op. (define as input when running session)
# tf Graph input

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

#Define Operations
add = tf.add(a,b)
mul = tf.multiply(a,b)

with tf.Session() as sess:
    print("Add by graph: ",sess.run(add,feed_dict={a:1,b:2}))
    print("Mul by graph: ",sess.run(mul,feed_dict={a:15,b:4}))

########################################################################################################################

# Create a Constant op that produces a  is
# added as a node to the default graph.1x2 matrix.  The op
#
# The value returned by the constructor represents the output
# of the Constant op.

matrix1 = tf.constant([[3.0,3.1]])

# Create another Constant that produces a 2x1 matrix.

matrix2 = tf.constant([[4.1],[4.0]])

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.

matrixMul = tf.matmul(matrix1,matrix2)

sess = tf.Session()
print("MatMul: ", sess.run(matrixMul))

