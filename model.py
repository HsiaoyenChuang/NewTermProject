import functools
import os
import sets
import tensorflow as tf

from config import *
MARGIN = batch_size * 2 + 0.2

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

class GRUModel:
    def __init__(self, ic, iq, ir, iw, state, dropout, num_hidden=400, num_layers=3):
        self.ic = ic # Input_c: context
        self.iq = iq # Input_q: question
        self.ir = ir # Input_r: right answer
        self.iw = iw # Input_w: wrong answer
        self.state = state # Encoded context & question from previous step
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.optimize
        self.evaluate

    @lazy_property
    def prediction(self):
        '''return embedded context & question vector, merged into one output'''
        self.cell = tf.nn.rnn_cell.GRUCell(self._num_hidden)
        self.network = tf.nn.rnn_cell.MultiRNNCell([self.cell] * self._num_layers)
        # embed context & question
        with tf.variable_scope('iq'):
            oq, _ = tf.nn.dynamic_rnn(self.network, self.iq, dtype=tf.float32)
            oq = tf.transpose(oq, [1, 0, 2])
        with tf.variable_scope('ic'):
            oc, _ = tf.nn.dynamic_rnn(self.network, self.ic, dtype=tf.float32)
            oc = tf.transpose(oc, [1, 0, 2])
        # select last output
        lq = tf.gather(oq, int(oq.get_shape()[0]) - 1)
        lc = tf.gather(oc, int(oc.get_shape()[0]) - 1)
        # combine embedding for question & context
        weight, bias = self._weight_and_bias(
            self._num_hidden, int(self.ir.get_shape()[1]))
        return lq + tf.matmul(lc,  weight) + bias

    @lazy_property
    def answer(self):
        '''return embedded right & wrong answer respectively'''
        self.cell = tf.nn.rnn_cell.GRUCell(self._num_hidden)
        self.network = tf.nn.rnn_cell.MultiRNNCell([self.cell] * self._num_layers)
        with tf.variable_scope('ir'):
            oa, _ = tf.nn.dynamic_rnn(self.network, self.ir, dtype=tf.float32)
            oa = tf.transpose(oa, [1, 0, 2])
        with tf.variable_scope('iw'):
            ow, _ = tf.nn.dynamic_rnn(self.network, self.iw, dtype=tf.float32)
            ow = tf.transpose(ow, [1, 0, 2])
        # select last output
        la = tf.gather(oa, int(oa.get_shape()[0]) - 1)
        lw = tf.gather(ow, int(ow.get_shape()[0]) - 1)
        return la, lw

    @lazy_property
    def cosine_cost(self):
        '''cosine distance as cost function'''
        r, w = self.answer
        return tf.reduce_mean(tf.maximum( # normailize with batch_size so don't have to change learning rate 
            0., MARGIN - self.cos_sim(self.state, r) + self.cos_sim(self.state, w))) / batch_size * 10

    @lazy_property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(1e-3)
        return optimizer.minimize(self.cosine_cost)

    @lazy_property
    def evaluate(self):
        '''evaluate cosine similarity of an answer option'''
        opt1, opt2 = self.answer
        # return self.cos_sim(self.state, opt1) - self.cos_sim(self.state, opt2)
        return self.cos_sim(self.state, opt1) / batch_size * 10 - self.cos_sim(self.state, opt2) / batch_size * 10

    def cos_sim(self, x, y):
        '''cosine similarity between 2D tensors x, y, both shape [n x m]'''
        assert x.get_shape().ndims == y.get_shape().ndims == 2, 'Must be 2D tensors'
        def l2_norm(x):
            norm = tf.sqrt(tf.reduce_sum(tf.square(x), keep_dims=True, reduction_indices=1))
            return tf.div(x, norm)
        x = l2_norm(x); y = l2_norm(y)
        return tf.reduce_sum(tf.mul(x, y))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    def save(self, sess, save_dir, dataset):
        self.saver = tf.train.Saver()
        print(" [*] Saving checkpoints (%s)..." % dataset)
        save_dir = os.path.join(save_dir, dataset)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.saver.save(sess, 
            os.path.join(save_dir, dataset))
        print(" [*] Checkpoints saved...")

    def load(self, sess, save_dir, dataset):
        self.saver = tf.train.Saver()
        print(" [*] Loading checkpoints (%s)..." % dataset)
        save_dir = os.path.join(save_dir, dataset)
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(save_dir, ckpt_name))
            print(" [*] Checkpoints loaded...")
            return True
        else:
            print(" [*] Checkpoints load failed...")
            return False
