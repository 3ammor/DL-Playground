import tensorflow as tf
from layers import *


class multilayer_perceptron:
    def __init__(self, config):
        self.config = config

        self.X = tf.placeholder(tf.float32, [None, n_input], name='X')
        self.y = tf.placeholder(tf.float32, [None, n_classes], name='y')

    def build_model(self):
        pass
