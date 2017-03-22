# coding=utf-8

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# CNN-01
# 乱数のシード
np.random.seed(20160704)
tf.set_random_seed(20160704)

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# CNN-03
# CNNを定義していく
num_filters1 = 32

x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, num_filters1],
                                          stddev=0.1))
h_conv1 = tf.nn.conv2d(x_image, w_conv1,
                       strides=[1, 1, 1, 1], padding='SAME')

# カットする値を決定する。初期値は0.1として学習する
b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_filters1]))
h_conv1_cutoff = tf.nn.relu(h_conv1 + b_conv1)

h_pool1 = tf.nn.max_pool(h_conv1_cutoff, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME')

# CNN-04
# 2段目の畳み込みフィルターとプーリング層
num_filters2 = 64

w_conv2 = tf.Variable(tf.truncated_normal([5, 5, num_filters1, num_filters2],
                                          stddev=0.1))
h_conv2 = tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME')

b_conv2 = tf.Variable(tf.constant(0.1, shape=[num_filters2]))
h_conv2_cutoff = tf.nn.relu(h_conv2 + b_conv2)

h_pool2 = tf.nn.max_pool(h_conv2_cutoff, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME')

# CNN-05
# 全結合層、ドロップアウト層、ソフトマックス関数の定義
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*num_filters2])

# 全結合層に入力するデータ数と全結合層のノード数
# 7*7サイズの画像データが64個全結合層に入力される
num_units1 = 7*7*num_filters2
num_units2 = 1024

w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
b2 = tf.Variable(tf.constant(0.1, shape=[num_units2]))
hidden2 = tf.nn.relu(tf.matmul(h_pool2_flat, w2) + b2)

# ドロップアウト層の処理
keep_prob = tf.placeholder(tf.float32)
hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

w0 = tf.Variable(tf.zeros([num_units2, 10]))
b0 = tf.Variable(tf.zeros([10]))
p = tf.nn.softmax(tf.matmul(hidden2_drop, w0) + b0)

# CNN-06
# 誤差関数、トレーニングアルゴリズム、正解率の定義
t = tf.placeholder(tf.float32, [None, 10])
loss = -tf.reduce_sum(t * tf.log(p))
# ネットワークが複雑になるほど、パラメータの値をより高精度に
# 最適化することが可能になるが、その分学習率を小さな値にしないと
# いけない。
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# CNN-07
# セッションの用意と保存
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
# セッションの状態を保存
# saver = tf.train.Saver()

# CNN-08
# パラメータの最適化
i = 0
for _ in range(20000):
    i += 1
    # 一回あたり50個のデータを使用
    batch_xs, batch_ts = mnist.train.next_batch(10)
    # 一回の最適化処理を行う部分
    # keep_prob 0.5 で全結合層からの出力を半分に
    sess.run(train_step,
             feed_dict={x: batch_xs, t: batch_ts, keep_prob: 0.5})

    # 500回ごとに正解率を表示
    if i % 500 == 0:
        loss_vals, acc_vals = [], []
        # メモリの使用量を減らすために、評価は4回に分けて実施
        for c in range(4):
            start = int(len(mnist.test.labels) / 4 * c)
            end = int(len(mnist.test.labels) / 4 * (c + 1))
            # 全結合層からの出力は切断しないように
            loss_val, acc_val = sess.run([loss, accuracy],
                                         feed_dict={x: mnist.test.images[start: end],
                                                    t: mnist.test.labels[start: end],
                                                    keep_prob: 1.0})
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
        loss_val = np.sum(loss_vals)
        acc_val = np.mean(acc_vals)
        print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))

        #saver.save(sess, 'cnn_session', global_step=i)

