import tensorflow as tf


# 데이터셋 x 와 y 를 placeholoder 노드로 선언
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# 우리의 모델(hypothesis)에서 W와 b를 텐서플로우 안에 trainable 변수로 선언
W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'basis')

# simple_linear_regression 의 hypothesis
hypothesis = W * x + b

# cost fucntion(loss function)
cost = tf.reduce_mean(tf.square(hypothesis -  y))
# 코스트를 최소화하기 위한 optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)

# 학습노드은 train 생성, optimizer를 이용해 cost(loss)를 줄이는것이 목표!
train = optimizer.minimize(cost)

# tensorflow 세션 생성, 실행(?)
sess = tf.Session()

# tensorflow 안에서 쓰이는 trainable variable을 먼저 세션 안에 선언(초기화?) 해준다
sess.run(tf.global_variables_initializer())

# 2000번 동안 train노드 실행한다. 데이터셋 x와 y는 placeholder 이므로 여기서 데이터를 넣어준다
for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train] , feed_dict={x: [1,2,3,4,5], y:[2.1, 3.1, 4.1, 5.1, 6.1]})

    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

# 위에서 학습시킨(2000번 반복을 통해) 모델을 확인, hypothesis변수는 우리가 trainable변수로 생성된 W와 b로 정의되어 있다.
print(sess.run(hypothesis, feed_dict={x: [5]}))
print(sess.run(hypothesis, feed_dict={x: [52.5]}))
print(sess.run(hypothesis, feed_dict={x: [3.5, 288]}))



