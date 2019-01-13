import tensorflow as tf

# data set 변수가 3개
x1_data = [73., 93., 89., 96., 73]
x2_data = [80., 88., 91., 98., 66]
x3_data = [75., 93., 90., 100., 70]
y_data = [152., 185., 180., 196., 142]

x_data = [[73., 80., 75.], [93., 88., 93.], [89., 91., 90.], [96., 98., 100.], [73., 66., 70]]
y_data = [[152.],[185.],[180.],[196.],[142.]]

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W)+ b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
train = optimizer.minimize(cost)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(10001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_data, Y:y_data})

    if step % 20 == 0:
        print(step, 'Cost: ', cost_val, 'Prediction: ', hy_val)
