import tensorflow as tf
from Tensorflow.base_trainer import BaseTrainer
from tensorflow.examples.tutorials.mnist import input_data


class MultilayerPerceptronTrainer(BaseTrainer):
    def __init__(self, sess, model, flags):
        super(MultilayerPerceptronTrainer, self).__init__(sess, model.config, flags)

        self.sess = sess
        self.model = model
        self.mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    def train(self):

        while self.cur_epoch_tensor.eval(self.sess) < self.model.config.training_epochs:
            self.sess.run(self.cur_epoch_assign_op, {self.cur_epoch_input: self.cur_epoch_tensor.eval(self.sess) + 1})

            avg_cost = 0.
            total_batch = int(self.mnist.train.num_examples / self.model.config.batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = self.mnist.train.next_batch(self.model.config.batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, accuracy = self.sess.run([self.model.optimizer, self.model.loss, self.model.accuracy],
                                               feed_dict={self.model.X: batch_x,
                                                          self.model.y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch

            # Display logs per epoch step
            tf.logging.log_every_n(
                tf.logging.INFO,
                'Epoch: {:04d},\t\t\t'.format(self.cur_epoch_tensor.eval(self.sess)) + 'cost= {:.9f}\t\t\t'.format(
                    avg_cost) + 'train accuracy= {:.5f}'.format(accuracy),
                self.model.config.display_step
            )

            if self.cur_epoch_tensor.eval(self.sess) % self.model.config.test_every == 0:
                self.test()

            summary_dic = {'loss': avg_cost}
            self.add_summary(self.cur_epoch_tensor.eval(self.sess), summary_dic)

            self.save()

        tf.logging.info('Finished training')

    def test(self):
        tf.logging.info("Test Accuracy: {:.5f}".format(self.model.accuracy.eval(session=self.sess,
                                                                                feed_dict={
                                                                                    self.model.X: self.mnist.test.images,
                                                                                    self.model.y: self.mnist.test.labels})))
