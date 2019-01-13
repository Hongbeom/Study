import tensorflow as tf
import numpy as np

tf.set_random_seed(777)
learning_rate = 0.01

x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[1], [1], [0], [0]]

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

X = tf.placeholder(shape=[None, 2], dtype=tf.float32)
Y = tf.placeholder(shape=[None, 1], dtype=tf.float32)

with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
    b1 = tf.Variable(tf.random_normal([2]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1)+b1)

    W1_hist = tf.summary.histogram('W1', W1)
    b1_hist = tf.summary.histogram('b1', b1)
    layer1_hist = tf.summary.histogram('layer1', layer1)

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
    b2 = tf.Variable(tf.random_normal([1]), name='bias2')
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    W2_hist = tf.summary.histogram('W2', W2)
    b2_hist = tf.summary.histogram('b2', b2)
    hypothesis_hist = tf.summary.histogram('hypothesis', hypothesis)

with tf.name_scope('cost'):
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

    cost_summ = tf.summary.scalar('cost', cost)

with tf.name_scope('train'):
    train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
accuracy_sum = tf.summary.scalar("accuracy", accuracy)

with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/xor_original")
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        summary, _ = sess.run([merged_summary, train], feed_dict={X:x_data, Y:y_data})
        writer.add_summary(summary, global_step=step)

        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)





