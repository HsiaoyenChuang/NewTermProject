# =================================================
# Start training a complete new model.
# And save it after ctrl-c.
# ==================================================
import tensorflow as tf
import numpy as np
import preprocess
from model import GRUModel
from tensorflow.contrib import learn

from config import *

# Data Preparatopn
# ==================================================
# Load data
print("Loading text data...")
qq, ll, cc, aa = preprocess.train_data(dataset)
c = [y for x in cc for y in x]
max_len = max([len(x) for x in c])
vocab = learn.preprocessing.VocabularyProcessor(max_len)
try:
    vocab.restore('save/vocab.pkl')
except:
    print("Building vocabularies...")
    vocab.fit(c)
    vocab.save('save/vocab.pkl')

print("Converting text to data...")
train_q = preprocess.build_vocab(qq, vocab)
train_c = [preprocess.build_vocab(x, vocab) for x in cc]
train_l = [preprocess.build_vocab(x, vocab) for x in ll]
train_a = [train_l[i][x-1] for i, x in enumerate(aa)]
# Shuffle data
def shuf_data(data):
    ''' Return shuf_data, [context, question, right_ans, wrong_ans]'''
    print("Shuffle data...")
    shuf_idx = np.random.permutation(np.arange(len(data)))
    data_shuf = [data[i] for i in shuf_idx]
    return data_shuf
def generate_dataset():
    ''''[[c, q, r, w, c_sents], ...], c is single sentence from c_sents'''
    result = []
    for i, c in enumerate(train_c):
        r = train_a[i]
        q = train_q[i]
        for w in train_l[i]:
            if (r == w).all():
                continue
            result.append([c, q, r, w])
    return [[y, x[1], x[2], x[3], x[0]] for x in result for y in x[0]]
# Training data
train_data = shuf_data(generate_dataset())
assert len(train_data) > 0
assert len(train_data[0]) == 5
ic, iq, ir, iw, ic_sents = zip(*train_data)
assert len(ic) == len(iq) == len(ir) == len(iw) == len(train_data)

print(" [*] Max sentence length: {:d}".format(max_len))
print(" [*] Vocabulary Size: {:d}".format(len(vocab.vocabulary_)))
print(" [*] Train Question Size: {:d}".format(len(train_data)))
# Build Model
# ==================================================
print("\nBuilding Model...")
row_size, rows = max_len, 1
input_c = tf.placeholder(tf.float32, [None, rows, row_size], name="ic")
input_q = tf.placeholder(tf.float32, [None, rows, row_size], name="iq")
input_r = tf.placeholder(tf.float32, [None, rows, row_size], name="ir")
input_w = tf.placeholder(tf.float32, [None, rows, row_size], name="iw")
state = tf.placeholder(tf.float32, [None, row_size], name="state")
dropout = tf.placeholder(tf.float32, name="dropout")
model = GRUModel(input_c, input_q, input_r, input_w, state, dropout, num_hidden=max_len)
# Train Model
# ==================================================
def encode(c_batch, q_batch):
    def merge(article, question):
        prev = zero_state
        for sent in article:
            prev = sess.run(model.prediction, {
                input_c: [sent], # 1 x [rows x row_size]
                input_q: [question], # 1 x [rows x row_size]
                input_r: zero_input, input_w: zero_input, state: prev, dropout: 0})
            return prev
    assert len(c_batch) == len(q_batch), "Must input same bacth size of context and question"
    encode_batch = [merge(c, q) for c, q in zip(c_batch, q_batch)]
    return encode_batch

costs = []
# fill placeholders with random data when they are not computed
zero_input = [np.random.randn(rows, row_size) for _ in range(batch_size)]
zero_state = [np.random.randn(row_size) for _ in range(batch_size)]

print("Running Model...")
print(" [*] Drop Probability: %s" % drop_prob)
print(" [*] Batch Size: %s" % batch_size)
print(" [*] Max Epoch: %s\n" % max_epoch)

sess = tf.Session()
try:
    sess.run(tf.global_variables_initializer())
except:
    sess.run(tf.initialize_all_variables())
print("Initialized. Start training...")

try:
    for epoch in range(max_epoch):
        for _ in range(100):
            # choose random question pairs
            idx = np.random.randint(max_len)
            # generate batches
            batch_iq = [[iq[idx]]] * batch_size # Input iq: batch_size x [1 x max_len]
            batch_ir = [[ir[idx]]] * batch_size # Input ir: batch_size x [max_len]
            batch_iw = [[iw[idx]]] * batch_size # Input iw: batch_size x [max_len]
            c_batch = [ic_sents[idx]] * batch_size # batch_size x [? x [max_len]]
            c_batch = [[[y] for y in x] for x in c_batch] # batch_size x [? x [1 x max_len]]
            # encode context & question for all context sentences
            batch_enc = encode(c_batch, batch_iq)
            # run training procedure
            sess.run(model.optimize, {
                # because batch is generated on same data, each batch shares same state
                input_c: zero_input, input_q: zero_input, input_r: batch_ir, input_w: batch_iw, state: batch_enc[0], dropout: drop_prob})
        # evaluate on training data
        error = sess.run(model.cosine_cost, {
            input_c: zero_input, input_q: zero_input, input_r: batch_ir, input_w: batch_iw, state: batch_enc[0], dropout: 1})
        costs.append(error)
        print('Epoch {:3d} cosine cost {:3.5f}, mean cost: {:3.5f}'.format(epoch + 1, error, sum(costs) / len(costs)))

except KeyboardInterrupt:
    pass

# ensure model is saved.
model.save(sess, save_dir='save', dataset=dataset)
