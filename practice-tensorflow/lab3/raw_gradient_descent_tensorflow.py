import tensorflow as tf

# data set
x_data = [1,2,3]
y_data= [1,2,3]

# W 를 trainable로 설정
W = tf.Variable(tf.random_normal([1]), name = "weight")
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# hypothesis
hypothesis =  W * X

# cost fucntion
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# learning rate = 0.1 이거를 왜 해주나면은 내가 생각하기에 기울기가 너무 큰값이 나오면은 계산을 할때 최저점을 지나칠 수 있으니, 상수(0.1)를 곱해서 천천히 최저점을 찾는것 같다!
# gradient = 경사, 기울기 즉 cost fucntion을 미분한것
learning_rate = 0.1

gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - (learning_rate * gradient)
update = W.assign(descent)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))


