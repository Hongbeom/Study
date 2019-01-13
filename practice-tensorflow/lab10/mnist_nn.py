import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_normal([784,256]), name='weight1')
    b1 = tf.Variable(tf.random_normal([256]), name='bias1')
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

    W1_hist = tf.summary.histogram('W1', W1)
    b1_hist = tf.summary.histogram('b1', b1)
    L1_hist = tf.summary.histogram('L1', L1)

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_normal([256, 256]), name='weight2')
    b2 = tf.Variable(tf.random_normal([256]), name='bias2')
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

    W2_hist = tf.summary.histogram('W2', W2)
    b2_hist = tf.summary.histogram('b2', b2)
    L2_hist = tf.summary.histogram('L2', L2)

with tf.name_scope('layer3'):
    W3 = tf.Variable(tf.random_normal([256, 10]), name='weight3')
    b3 = tf.Variable(tf.random_normal([10]), name='bias3')
    hypothesis = tf.matmul(L2, W3) + b3

    W3_hist = tf.summary.histogram('W3', W3)
    b3_hist = tf.summary.histogram('b3', b3)
    hypothesis_hist = tf.summary.histogram('hypothesis', hypothesis)

with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))

    cost_summ = tf.summary.scalar('cost', cost)

with tf.name_scope('train'):
    train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    accuracy_summ = tf.summary.scalar('accuracy', accuracy)


with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./MNIST/mnist_nn_simple')
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())
    step = -1
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            feed_dict = {X:batch_xs, Y:batch_ys}
            c, _, summary = sess.run([cost, train, merged_summary], feed_dict)
            avg_cost += c / total_batch
            step += 1
            writer.add_summary(summary, global_step=step)
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')

    # Test model and check accuracy
    print('Accuracy:', sess.run(accuracy, feed_dict={
        X: mnist.test.images, Y: mnist.test.labels}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))