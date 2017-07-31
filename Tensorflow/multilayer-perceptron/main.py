import tensorflow as tf
from model import MultilayerPerceptron
from Tensorflow.config import multilayer_perceptron_config
from Tensorflow.utils import create_dirs
from trainer import MultilayerPerceptronTrainer

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', "/tmp/multilayer-perceptron/", """ Experiment dir to store ckpt & summaries """)
tf.app.flags.DEFINE_string('summary_dir', "/tmp/multilayer-perceptron/summaries", """ Experiment dir to store ckpt & summaries """)
tf.app.flags.DEFINE_boolean('is_train', True, """ Whether it is a training or testing""")
tf.app.flags.DEFINE_boolean('cont_train', True, """ whether to Load the Model and Continue Training or not """)
tf.app.flags.DEFINE_boolean('train_n_test', True, """ whether to Load the Model and Continue Training or not """)


def main(_):
    create_dirs([FLAGS.summary_dir])

    config = multilayer_perceptron_config()
    model = MultilayerPerceptron(config)

    sess = tf.Session()

    trainer = MultilayerPerceptronTrainer(sess, model, FLAGS)

    trainer.train()


if __name__ == '__main__':
    tf.app.run()
