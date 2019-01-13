import numpy as np
import tensorflow as tf

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
# 레이블은 총 7개 0~6
nb_classes = 7

X = tf.placeholder(tf.float32, shape=[None, 16])
# Y과 웬일로 int32: 아마 레이블이니까 정수밖에 없어서 그렇게 해준거 같음!
Y = tf.placeholder(tf.int32, shape=[None, 1])
# one_hot 함수로 데이터를 one_hot 코딩형식으로 변환해줌. 하지만 차원이 하나가 늘어남
# example: [None, 1] --> one_hot --> [None, 1, nb_classes]
Y_one_hot = tf.one_hot(Y, nb_classes)
# 그러므로 reshape 을 사용합니다
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# softmax 를 좀더 편하게 쓰자
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)

cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.arg_max(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2000):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X:x_data, Y:y_data})
            print("Step: {:5}\tLoss: {:3f}\tAcc: {:2%}".format(step, loss, acc))
    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X:x_data})
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format( p == int(y), p, int(y)))





