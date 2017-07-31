import tensorflow as tf
from model import ConvolutionalNeuralNetwork
from Tensorflow.config import convolutional_neural_network_config
from Tensorflow.utils import create_dirs
from trainer import ConvolutionalNeuralNetworkTrainer

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', "/tmp/convolutional-neural-network/", """ Experiment dir to store ckpt & summaries """)
tf.app.flags.DEFINE_string('summary_dir', "/tmp/convolutional-neural-network/summaries", """ Experiment dir to store ckpt & summaries """)
tf.app.flags.DEFINE_boolean('is_train', True, """ Whether it is a training or testing""")
tf.app.flags.DEFINE_boolean('cont_train', True, """ whether to Load the Model and Continue Training or not """)
tf.app.flags.DEFINE_boolean('train_n_test', True, """ whether to Load the Model and Continue Training or not """)


def main(_):
    create_dirs([FLAGS.summary_dir])

    config = convolutional_neural_network_config()
    model = ConvolutionalNeuralNetwork(config)

    sess = tf.Session()

    trainer = ConvolutionalNeuralNetworkTrainer(sess, model, FLAGS)

    trainer.train()


if __name__ == '__main__':
    tf.app.run()
