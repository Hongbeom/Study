import tensorflow as tf

X = [1,2,3]
Y = [1,2,3]



W = tf.Variable(5.0)

hypothesis = X*W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
gradient = tf.reduce_mean((W*X - Y )*X) * 2 ## 왜 여기다 2를 곱해줄까... 아마도 optimizer 안에서 미분으로 기울기를 계산을 때릴때 상수를 생강을 안해줘서 그러는거 같다. 여기는 2차식이기 때문에 2를 곱해준거 같습니다.

# Get gradients
gvs = optimizer.compute_gradients(cost)

# 요기 안에서 trainable variable(W) 나 기울(descent) 조금 변형시키거나 수정 해서 optimizer에 적용시켜 학습을 진행시킨다!
apply_gradients = optimizer.apply_gradients(gvs)


sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(100):
    print(i, sess.run([gradient, W, gvs]))
    sess.run(apply_gradients)