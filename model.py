import functools
import sets
import tensorflow as tf

MARGIN = 20000

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
    def __init__(self, ic, iq, ir, iw, dropout, num_hidden=400, num_layers=3):
        self.ic = ic # Input_c: context
        self.iq = iq # Input_q: question
        self.ir = ir # Input_r: right answer
        self.iw = iw # Input_w: wrong answer
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
    def encode_answer(self):
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
        emb = self.prediction
        r,w = self.encode_answer
        return tf.reduce_mean(
            tf.maximum(0., MARGIN - self.cos_sim(emb, r) + self.cos_sim(emb, w)))

    @lazy_property
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(1e-3)
        return optimizer.minimize(self.cosine_cost)

    @lazy_property
    def evaluate(self):
        return self.cosine_cost

    def cos_sim(self, x, y):
        '''Return cosine similarity between 2D tensors x, y, both shape [n x m]'''
        return tf.reduce_sum(tf.matmul(x, y, transpose_b=True))

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

