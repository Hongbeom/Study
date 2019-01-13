import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(5.0)

hypothesis = X*W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

minimize = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(21):
    sess.run(minimize, feed_dict={X:x_data, Y:y_data})
    print(i, sess.run(W), sess.run(cost, feed_dict={X:x_data, Y:y_data}))