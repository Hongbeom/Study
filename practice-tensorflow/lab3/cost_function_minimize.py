import tensorflow as tf
import matplotlib.pyplot as plt

# 아주 간단한 데이터 셋
X = [1,2,3]
Y = [1,2,3]

# W 값 >> placeholder 로 줍니다.
W = tf.placeholder(tf.float32)

# hypothesis
hypothesis = W * X

# cost fucntion
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 세션 초기화
sess = tf.Session()

# variable들을 세션안에 넣어준다 라는 느낌
sess.run(tf.global_variables_initializer())

# variables for plotting cost fucntion
W_val = []
cost_val = []


for i in range(-30, 50):
    feed_W = i * 0.1  # 여기서 range를 -30부터 50으로 해주고 10으로 나눠준 이유는 그래프를 좀더 부드러운 곡선으로
                      # -3 과 5 사이의 소수점 첫재짜리까지 포함하기 위해서 그래준거 같다!
                      # 러닝 어쩌고 저쩌고 상수( 알파) 와 관련이 읎다
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

# show the cost fucntion
plt.plot(W_val, cost_val)
plt.show()







