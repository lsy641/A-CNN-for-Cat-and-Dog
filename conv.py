#coding=utf-8
import tensorflow as tf
def initialize_parameters():
    tf.set_random_seed(1)
    W1 = tf.get_variable("W1", [3, 3, 3, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [3, 3, 16, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters = {"W1": W1,
                  "W2": W2}
    return parameters

def inference(images, batch_size, n_classes,parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    with tf.variable_scope('conv1') as scope:
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, W1, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.variable_scope('pooling1_lrn') as scope:
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling1')
            norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    with tf.variable_scope('conv2') as scope:
                biases = tf.get_variable('biases',
                                         shape=[16],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.1))
                conv = tf.nn.conv2d(norm1, W2, strides=[1, 1, 1, 1], padding='SAME')
                pre_activation = tf.nn.bias_add(conv, biases)
                conv2 = tf.nn.relu(pre_activation, name='conv2')
    # pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pooling2')
    with tf.variable_scope('local3') as scope:
        local3 = tf.reshape(pool2, shape=[batch_size, -1])
        local3 = tf.contrib.layers.fully_connected(local3, 128, activation_fn=None)
        #     reshape = tf.reshape(pool2, shape=[batch_size, -1])
        #     dim = reshape.get_shape()[1].value
        #     weights = tf.get_variable('weights',
        #                               shape=[dim, 128],
        #                               dtype=tf.float32,
        #                               initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        #     biases = tf.get_variable('biases',
        #                              shape=[128],
        #                              dtype=tf.float32,
        #                              initializer=tf.constant_initializer(0.1))
        # local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

        # local4
    with tf.variable_scope('local4') as scope:
        # weights = tf.get_variable('weights',
        #                           shape=[128, 128],
        #                           dtype=tf.float32,
        #                           initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        # biases = tf.get_variable('biases',
        #                          shape=[128],
        #                          dtype=tf.float32,
        #                          initializer=tf.constant_initializer(0.1))
        # local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')
        local4 = tf.reshape(local3, shape=[batch_size, -1])
        local4 = tf.contrib.layers.fully_connected(local4, 128, activation_fn=None)

    with tf.variable_scope('softmax_linear') as scope:
        softmax_linear = tf.reshape(local4, shape=[batch_size, -1])
        softmax_linear = tf.contrib.layers.fully_connected(softmax_linear, 2, activation_fn=None)
    return softmax_linear



def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
                        (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss

def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op

def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy