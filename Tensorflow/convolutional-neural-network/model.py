import tensorflow as tf


class ConvolutionalNeuralNetwork:
    def __init__(self, config):
        self.config = config

        self.X = tf.placeholder(tf.float32, [None, self.config.n_input], name='X')
        self.y = tf.placeholder(tf.float32, [None, self.config.n_classes], name='y')
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

        with tf.variable_scope('hidden_layer'):
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

        with tf.variable_scope('output_layer'):
            output = tf.layers.conv2d(
                drp2,
                self.config.n_classes,
                7,
                padding='valid',
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='conv'
            )
            self.output = tf.squeeze(output)

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
            # Calculate accuracy
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
