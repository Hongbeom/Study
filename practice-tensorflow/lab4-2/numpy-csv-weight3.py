import numpy as np
import tensorflow as tf

xy = np.loadtxt('exam.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, :-1]
y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, shape = [None, 3])
Y = tf.placeholder(tf.float32, shape = [None, 1])
W = tf.Variable(tf.random_normal([3, 1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)

train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(10000):
    cost_val, hy_val, _, W_val, b_val= sess.run([cost, hypothesis, train, W, b], feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print(step, cost_val, hy_val, W_val, b_val)

print('Your score is about: ', sess.run(hypothesis, feed_dict={X:[[100, 70, 101]]}))
print('Others score is about: ', sess.run(hypothesis, feed_dict={X:[[60, 70, 110], [90, 100, 80]]}))