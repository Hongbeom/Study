# inflearn - lab 2
import tensorflow as tf
# X and Y data

x_train = [1,2,3]
y_train = [1,2,3]

# 텐서플로우 안에서 학습하는 과정에서 변경시켜주는 변수들! >> trainable!
W = tf.Variable(tf.random_normal([1]), name = 'Weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# hypothesis 노드
hypotehsis = x_train * W + b

# cost fucntion
cost = tf.reduce_mean(tf.square(hypotehsis - y_train))
# reudce_mean은 평균을 내주는 것!

# 코스트 최소화 시키기! -- GradientDescent 로 optimizer 선언, optimizer 안의 minimize 로 코스트를 최소화 하는게 목표!
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

# 세션 초기화
sess = tf.Session()

# Initializes global variables in the graph - 텐서플로우 variables을 전역으로 처리해준다(초기화 느낌, 여기서는 W 와 b)
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train) # cost를 최소화하는 train노드를 실행시킨다! - 학습의 시작!
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(hypotehsis),sess.run(W), sess.run(b))