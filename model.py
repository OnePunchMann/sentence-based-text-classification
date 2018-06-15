import tensorflow as tf
from data_helpers import batch_maker
from data_helpers import data_provider


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shae):
    initial = tf.constant(0.1, shape=shae)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool(x, word):
    return tf.nn.max_pool(x, ksize=[1, word - 2, 1, 1], strides=[1, 1, 1, 1], padding='VALID')


def CNN():
    # 一次选择单词数
    filtersize = 3
    # 句子的最大词数
    # 词向量的维度
    word = 50
    vector = 100
    batch_size = 50

    x = tf.placeholder(tf.float32, [None, word * vector])
    y_ = tf.placeholder(tf.float32, [None, 2])
    x_sentence = tf.reshape(x, [-1, word, vector, 1])

    sess = tf.InteractiveSession()
    w_conv1 = weight_variable([filtersize, vector, 1, 3])
    b_conv1 = bias_variable([3])

    h_conv1 = tf.nn.relu(conv2d(x_sentence, w_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1, word)

    w_fc1 = weight_variable([word * 1 * 3, 2])
    b_fc1 = bias_variable([2])

    h_pool1_flat = tf.reshape(h_pool1, [-1, word * 1 * 3])

    y_conv = tf.nn.softmax(tf.matmul(h_pool1_flat, w_fc1) + b_fc1)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.global_variables_initializer().run()
    for i in range(1000):
        data_x, data_y = data_provider()
        batch_x, batch_y = batch_maker(batch_size, 50, data_x, data_y)
        if i % 10 == 0:
            train_accurary = accuracy.eval(feed_dict={x: batch_x, y_: batch_y})
            print("step %d,training accuracy %g" % (i, train_accurary))
        train_step.run(feed_dict={x: batch_x, y_: batch_y})


# 下面这句话是在测试集上的测试，但是还没分出测试集。。。
# print("test accuracy %g"%accuracy.eval(feed_dict={x:batch_x,y_:batch_y}))
if __name__ == '__main__':
    CNN()
