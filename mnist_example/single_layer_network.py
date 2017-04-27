# coding=utf-8

import tensorflow as tf
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

# 乱数のシード
np.random.seed(20160612)
tf.set_random_seed(20160612)

mnist = input_data.read_data_sets("/tmp/tensorflow/data/", one_hot=True)


class SingleLayerNetwork:
    """
    中間層が1層のニューラルネットワーク
    """

    # 層の数
    NUM_UNITS = 1024

    def __init__(self):
        """
        コンストラクタ
        グラフコンテキストを開始
        """
        with tf.Graph().as_default():
            self.prepare_model()
            self.prepare_session()

    def prepare_model(self):
        """
        モデルの作成
        """
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 784], name='input')

        with tf.name_scope('hidden'):
            # 隠れ層の出力を計算
            w1 = tf.Variable(tf.truncated_normal([784, self.NUM_UNITS]),
                             name='weights')
            b1 = tf.Variable(tf.zeros([self.NUM_UNITS]), name='biases')
            hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1, name='hidden1')

        with tf.name_scope('output'):
            # ソフトマックス関数で確率を計算
            w0 = tf.Variable(tf.zeros([self.NUM_UNITS, 10]), name='weights')
            b0 = tf.Variable(tf.zeros([10]), name='biases')
            p = tf.nn.softmax(tf.matmul(hidden1, w0) + b0, name='softmax')

        with tf.name_scope('optimizer'):
            t = tf.placeholder(tf.float32, [None, 10], name='labels')
            # 誤差関数
            loss = -tf.reduce_sum(t * tf.log(p), name='loss')
            # トレーニングアルゴリズム
            train_step = tf.train.AdamOptimizer().minimize(loss)

        with tf.name_scope('evaluator'):
            correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
            # 正解率
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
                                      name='accuracy')

        # 値の変化をグラフ表示する要素を宣言
        # 折れ線グラフ
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        # ヒストグラム
        tf.summary.histogram("weights_hidden", w1)
        tf.summary.histogram("biases_hidden", b1)
        tf.summary.histogram("weights_output", w0)
        tf.summary.histogram("biases_output", b0)

        # 外部から参照する必要のあるものをインスタンス変数として公開
        self.x, self.t, self.p = x, t, p
        self.train_step = train_step
        self.loss = loss
        self.accuracy = accuracy

    def prepare_session(self):
        """
        セッション用意
        データの保存 
        """
        # 変数の初期化
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        summary = tf.summary.merge_all()
        # TensorBoardデータの出力先
        writer = tf.summary.FileWriter("/tmp/tensorflow/mnist_sl_logs",
                                       sess.graph)

        # 学習データ保存の定義
        # saver = tf.train.Saver()

        # インスタンス変数として公開
        self.sess = sess
        self.summary = summary
        self.writer = writer


# 以下はセッションを用意してパラメータの最適化を実行

# インスタンスを作成
nn = SingleLayerNetwork()

# 開始時間
start_time = time.time()

# 勾配降下法でパラメータの最適化
i = 0
for _ in range(2000):
    i += 1
    # 取り出したデータを記憶しており、呼び出すごとに次のデータを取り出す。
    batch_xs, batch_ts = mnist.train.next_batch(100)
    # 勾配降下法によるパラメータの修正
    nn.sess.run(nn.train_step, feed_dict={nn.x: batch_xs, nn.t: batch_ts})
    # 100回ごとに、その時点のパラメータでテストセットに対する誤差関数
    # と正解率の値を計算
    if i % 100 == 0:
        summary, loss_val, acc_val = nn.sess.run(
                [nn.summary, nn.loss, nn.accuracy],
                feed_dict={nn.x: mnist.test.images, nn.t: mnist.test.labels})
        print('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))

        # TensorBoard用
        nn.writer.add_summary(summary, i)
        # 学習データのセーブ
        # nn.saver.save(nn.sess, '/tmp/tensorflow/saver/sln_session', global_step=i)

# 経過時間の表示
elapsed_time = time.time() - start_time
print(("elapsed_time:{0}".format(elapsed_time)) + "[sec]")