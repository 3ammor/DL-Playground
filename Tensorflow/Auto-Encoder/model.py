import tensorflow as tf


class AutoEncoder:
    def __init__(self, config):
        self.config = config

        self.X = tf.placeholder(tf.float32, [None, self.config.n_input], name='X')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        self.build_model()

    def build_model(self):
        with tf.variable_scope('input_layer'):
            X = tf.reshape(self.X, shape=self.config.input_shape)
            conv1 = tf.layers.conv2d(
                X,
                16,
                3,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='conv'
            )
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            drp1 = tf.layers.dropout(pool1, rate=self.config.dropout_rate, training=self.is_training, name='dropout')

        with tf.variable_scope('conv_layer'):
            conv2 = tf.layers.conv2d(
                drp1,
                32,
                3,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='conv'
            )
            pool1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            drp2 = tf.layers.dropout(pool1, rate=self.config.dropout_rate, training=self.is_training, name='dropout')

        with tf.variable_scope('deconv_layer'):
            deconv = tf.nn.conv2d_transpose(drp2, [3, 3, 1, 1],
                                            [-1, 14, 14, 1], [1, 2, 2, 1], padding='SAME', name='deconv')
            drp3 = tf.layers.dropout(deconv, rate=self.config.dropout_rate, training=self.is_training, name='dropout')

        with tf.variable_scope('output_layer'):
            self.output = tf.nn.conv2d_transpose(drp3, [3, 3, 1, 1],
                                                 [-1, 28, 28, 1], [1, 2, 2, 1], padding='SAME', name='deconv')

        with tf.name_scope('loss'):
            self.loss = tf.losses.mean_squared_error(self.X, self.output)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
