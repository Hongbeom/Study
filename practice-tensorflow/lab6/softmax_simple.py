import tensorflow as tf

x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 3])
# 레이블의 갯수
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='weight')

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predicted = tf.argmax(hypothesis, 1)
real = tf.argmax(Y, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, real), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, a, b = sess.run([train, cost, hypothesis], feed_dict={X:x_data, Y:y_data})
        if step % 200 == 0:
            print(step, a, b)

    d, e, f = sess.run([predicted, real, accuracy], feed_dict={X:x_data, Y:y_data})
    print(d,e,f)


