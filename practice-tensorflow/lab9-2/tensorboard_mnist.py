# 내가 보기에는 이거 이상한 코드임 ㅎㅎ
import tensorflow as tf
import random
import matplotlib.pyplot as plt
tf.set_random_seed(777)  # for reproducibility

from tensorflow.examples.tutorials.mnist import input_data
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes])

with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_normal([784, 50]), name='weight1')
    b1 = tf.Variable(tf.random_normal([50], name='bias1'))
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    W1_his = tf.summary.histogram('W1', W1)
    b1_his = tf.summary.histogram('b1', b1)
    layer1_his = tf.summary.histogram('layer1', layer1)

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_normal([50, 50]), name='weight2')
    b2 = tf.Variable(tf.random_normal([50], name='bias2'))
    layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    W2_his = tf.summary.histogram('W2', W2)
    b2_his = tf.summary.histogram('b2', b2)
    layer2_his = tf.summary.histogram('layer2', layer2)

with tf.name_scope('layer3'):
    W3 = tf.Variable(tf.random_normal([50, 50]), name='weight3')
    b3 = tf.Variable(tf.random_normal([50], name='bias3'))
    layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

    W3_his = tf.summary.histogram('W3', W3)
    b3_his = tf.summary.histogram('b3', b3)
    layer3_his = tf.summary.histogram('layer3', layer3)

with tf.name_scope('layer4'):
    W4 = tf.Variable(tf.random_normal([50, 50]), name='weight4')
    b4 = tf.Variable(tf.random_normal([50], name='bias4'))
    layer4 = tf.sigmoid(tf.matmul(layer3, W4) + b4)

    W4_his = tf.summary.histogram('W4', W4)
    b4_his = tf.summary.histogram('b4', b4)
    layer4_his = tf.summary.histogram('layer4', layer4)

with tf.name_scope('layer5'):
    W5 = tf.Variable(tf.random_normal([50, 50]), name='weight5')
    b5 = tf.Variable(tf.random_normal([50], name='bias5'))
    layer5 = tf.sigmoid(tf.matmul(layer4, W5) + b5)

    W5_his = tf.summary.histogram('W5', W5)
    b5_his = tf.summary.histogram('b5', b5)
    layer5_his = tf.summary.histogram('layer5', layer5)

with tf.name_scope('layer6'):
    W6 = tf.Variable(tf.random_normal([50, 10]), name='weight6')
    b6 = tf.Variable(tf.random_normal([10], name='bias6'))
    hypothesis = tf.nn.softmax(tf.matmul(layer5, W6) + b6)

    W6_his = tf.summary.histogram('W6', W6)
    b6_his = tf.summary.histogram('b6', b6)
    hypothesis_his = tf.summary.histogram('hypothesis', hypothesis)


with tf.name_scope('cost'):
    cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
    cost_summ = tf.summary.scalar('cost', cost)

with tf.name_scope('train'):
    train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
accuracy_summ = tf.summary.scalar('accuracy', accuracy)
# parameters - 1 epoch 은 전체 training data set을 한번 학습시키는것
training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./mnist_logs/mnist_soft_sig_DEEP')
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary, c, _ = sess.run([merged_summary, cost, train], feed_dict={X:batch_xs, Y:batch_ys})
            avg_cost += c / total_batch
            writer.add_summary(summary, global_step=(100 * epoch) + i)
        print('Epoch: ', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y:mnist.test.labels}))

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction:", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r+1]}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()

