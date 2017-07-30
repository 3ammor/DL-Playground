import tensorflow as tf


class MultilayerPerceptron:
    def __init__(self, config):
        self.config = config

        self.X = tf.placeholder(tf.float32, [None, self.config.n_input], name='X')
        self.y = tf.placeholder(tf.float32, [None, self.config.n_classes], name='y')

        self.build_model()

    def build_model(self):
        with tf.name_scope('input_layer'):
            h1 = tf.layers.dense(
                self.X,
                self.config.n_hidden_1,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='dense'
            )

        with tf.name_scope('hidden_layer'):
            h2 = tf.layers.dense(
                h1,
                self.config.n_hidden_2,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='dense'
            )

        with tf.name_scope('output_layer'):
            self.output = tf.layers.dense(
                h2,
                self.config.n_classes,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='dense'
            )

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
            # Calculate accuracy
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

