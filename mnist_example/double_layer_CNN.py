# coding=utf-8

import tensorflow as tf
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

# 乱数のシード
np.random.seed(20160704)
tf.set_random_seed(20160704)

mnist = input_data.read_data_sets("/tmp/tensorflow/mist_data/", one_hot=True)


class DoubleLayerCNN():
    """
    CNNを定義する
    モデルの作成
    """

    def __init__(self):
        """
        コンストラクタ
        フィルターの設定
        グラフコンテキストを開始
        """
        with tf.Graph().as_default():
            self.NUM_FILTERS1 = 32
            self.NUM_FILTERS2 = 64

            self.prepare_model()
            self.prepare_session()

    def prepare_model(self):
        """
        モデルの定義
        """
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 784])

        with tf.name_scope('pool1'):
            x_image = tf.reshape(x, [-1, 28, 28, 1])

            w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1,
                                                       self.NUM_FILTERS1],
                                                       stddev=0.1))
            h_conv1 = tf.nn.conv2d(x_image, w_conv1,
                                   strides=[1, 1, 1, 1], padding='SAME')

            # カットする値を決定する。初期値は0.1として学習する
            b_conv1 = tf.Variable(tf.constant(0.1, shape=[self.NUM_FILTERS1]))
            h_conv1_cutoff = tf.nn.relu(h_conv1 + b_conv1)

            h_pool1 = tf.nn.max_pool(h_conv1_cutoff, ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1], padding='SAME')

        with tf.name_scope('pool2'):
            # 2段目の畳み込みフィルターとプーリング層
            w_conv2 = tf.Variable(tf.truncated_normal([5, 5, self.NUM_FILTERS1,
                                                       self.NUM_FILTERS2],
                                                       stddev=0.1))
            h_conv2 = tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1],
                                   padding='SAME')

            b_conv2 = tf.Variable(tf.constant(0.1, shape=[self.NUM_FILTERS2]))
            h_conv2_cutoff = tf.nn.relu(h_conv2 + b_conv2)

            h_pool2 = tf.nn.max_pool(h_conv2_cutoff, ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1], padding='SAME')

        with tf.name_scope('hidden2'):
            # 全結合層、ドロップアウト層、ソフトマックス関数の定義
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*self.NUM_FILTERS2])

            # 全結合層に入力するデータ数と全結合層のノード数
            # 7*7サイズの画像データが64個全結合層に入力される
            NUM_UNITS1 = 7*7*self.NUM_FILTERS2
            NUM_UNITS2 = 1024

            w2 = tf.Variable(tf.truncated_normal([NUM_UNITS1, NUM_UNITS2]), name='weights')
            b2 = tf.Variable(tf.constant(0.1, shape=[NUM_UNITS2]), name='biases')
            hidden2 = tf.nn.relu(tf.matmul(h_pool2_flat, w2) + b2, name='hidden2')

        with tf.name_scope('dropout'):
            # ドロップアウト層の処理
            keep_prob = tf.placeholder(tf.float32)
            hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

        with tf.name_scope('output'):
            # ソフトマックス関数で確率を計算
            w0 = tf.Variable(tf.zeros([NUM_UNITS2, 10]), name='weights')
            b0 = tf.Variable(tf.zeros([10]), name='biases')
            p = tf.nn.softmax(tf.matmul(hidden2_drop, w0) + b0, name='softmax')

        with tf.name_scope('optimizer'):
            # 誤差関数、トレーニングアルゴリズム、正解率の定義
            t = tf.placeholder(tf.float32, [None, 10], name='labels')
            loss = -tf.reduce_sum(t * tf.log(p))
            # ネットワークが複雑になるほど、パラメータの値をより高精度に
            # 最適化することが可能になるが、その分学習率を小さな値にしないといけない。
            train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

        with tf.name_scope('evaluator'):
            correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
            # 正解率
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
                                      name='accuracy')

        # 値の変化をグラフ表示する要素を宣言。
        # 折れ線グラフ
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        # ヒストグラム

        tf.summary.scalar("weights_hidden2", w2)
        tf.summary.scalar("biases_hidden2", b2)

        tf.summary.histogram("weights_output", w0)
        tf.summary.histogram("biases_output", b0)

        # 外部から参照する必要のあるものをインスタンス変数として公開
        self.x, self.t, self.p = x, t, p
        self.train_step = train_step
        self.loss = loss
        self.accuracy = accuracy
        self.keep_prob = keep_prob

    def prepare_session(self):
        """
        セッションの用意
        データの保存 
        """
        # セッションの用意
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        summary = tf.summary.merge_all()
        # TensorBoardデータの出力先
        writer = tf.summary.FileWriter("/tmp/tensorflow/mnist_cnn_logs",
                                       sess.graph)
        # 学習データ保存の定義
        saver = tf.train.Saver()

        # インスタンス変数として公開
        self.sess = sess
        self.summary = summary
        self.writer = writer
        self.saver = saver


# インスタンスを作成
nn = DoubleLayerCNN()

# 開始時間
start_time = time.time()

# パラメータの最適化
i = 0
for _ in range(20000):
    i += 1
    # 一回あたり50個のデータを使用
    batch_xs, batch_ts = mnist.train.next_batch(50)
    # 一回の最適化処理を行う部分
    # keep_prob 0.5 で全結合層からの出力を半分に
    nn.sess.run(nn.train_step,
                feed_dict={nn.x: batch_xs, nn.t: batch_ts, nn.keep_prob: 0.5})

    # 500回ごとに正解率と誤差を表示
    if i % 500 == 0:
        # 全結合層からの出力は切断しないように
        summary, loss_val, acc_val = nn.sess.run([nn.summary, nn.loss, nn.accuracy],
                                                  feed_dict={nn.x: mnist.test.images,
                                                             nn.t: mnist.test.labels,
                                                             nn.keep_prob: 1.0})

        print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))

        # TensorBoard用
        nn.writer.add_summary(summary, i)
        # 学習データのセーブ
        nn.saver.save(nn.sess, '/tmp/tensorflow/saver/cnn_session', global_step=i)

# 経過時間の表示
elapsed_time = time.time() - start_time
print(("elapsed_time:{0}".format(elapsed_time)) + "[sec]")
