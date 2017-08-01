import tensorflow as tf


class RecurrentNeuralNetwork:
    def __init__(self, config):
        self.config = config

        self.X = tf.placeholder(tf.float32, [None, self.config.n_steps, self.config.n_input], name='X')
        self.y = tf.placeholder(tf.float32, [None, self.config.n_classes], name='y')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        self.build_model()

    def build_model(self):
        # with tf.variable_scope('input_layer'):
            # x = tf.unstack(self.X, self.config.n_steps, 1)

        with tf.variable_scope('lstm_layer'):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.config.n_hidden, forget_bias=1.0)
            outs, states = tf.nn.dynamic_rnn(lstm_cell, self.X, dtype=tf.float32)

        with tf.variable_scope('output_layer'):
            self.output = tf.layers.dense(
                outs[:, -1, :],
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
