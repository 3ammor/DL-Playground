import tensorflow as tf
from base_trainer import BaseTrainer
from tf.examples.tutorials.mnist import input_data


class MultilayerPerceptronTrainer(BaseTrainer):
    def __init__(self, sess, model, flags):
        super(MultilayerPerceptronTrainer, self).__init__(sess, model.config, flags)

        self.model = model
        self.mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


    def train(self):

        for epoch in range(config.training_epochs):
            avg_cost = 0.
            total_batch = int(self.mnist.train.num_examples / config.batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = self.mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([model.optimizer, model.loss], feed_dict={model.X: batch_x,
                                                                          model.Y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch

            # Display logs per epoch step
            tf.logging.log_every_n(
                tf.logging.INFO,
                "Epoch:", '%04d' % (epoch + 1), "cost=", \
                "{:.9f}".format(avg_cost),
                config.display_step,
                *args
            )

            summary_dic = {'loss': avg_cost}
            self.add_summary(epoch, summary_dic)

        tf.logging.info('Finished training')

    def test(self):
        tf.logging.info("Accuracy:", model.accuracy.eval({model.X: self.mnist.test.images, model.y: self.mnist.test.labels}))
