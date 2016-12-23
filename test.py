# ==================================================
# Load and test a trained model on Traing Set.
# ==================================================
import tensorflow as tf
import numpy as np
import preprocess
from model import GRUModel
from tensorflow.contrib import learn

from config import *

print("Restoring Vocab...")
test_data = preprocess.train_data(dataset)
c = [y for x in test_data[2] for y in x]
max_len = max([len(x) for x in c])
vocab = learn.preprocessing.VocabularyProcessor(max_len)
vocab.fit(c)

# 1/10 for testing
idx = np.random.randint(len(test_data[0]), size=int(len(test_data[0])/10))
qq, ll, cc, aa = zip(*[[test_data[0][i], test_data[1][i], test_data[2][i], test_data[3][i]] for i in idx])

truths = aa

print("Converting text to data...")
train_q = preprocess.build_vocab(qq, vocab)
train_c = [preprocess.build_vocab(x, vocab) for x in cc]
train_l = [preprocess.build_vocab(x, vocab) for x in ll]
train_a = [train_l[i][x-1] for i, x in enumerate(aa)]

def generate_dataset():
    result = []
    for i, c in enumerate(train_c):
        r = train_a[i]
        q = train_q[i]
        chioces = train_l[i]
        for w in train_l[i]:
            if (r == w).all():
                continue
            result.append([c, q, r, w, truths[i]])
    return [[y, x[1], x[2], x[3], x[0], chioces, x[4]] for x in result for y in x[0]]

test_data = generate_dataset()
idx = np.random.randint(len(test_data), size=len(qq))
test_data = [test_data[i] for i in idx]
ic, iq, ir, iw, ic_sents, chioces, truths = zip(*test_data)

print(" [*] Max sentence length: {:d}".format(max_len))
print(" [*] Vocabulary Size: {:d}".format(len(vocab.vocabulary_)))
print(" [*] Test Question Size: {:d}".format(len(test_data)))

# ==================================================
print("Restoring Model...")
input_c = tf.placeholder(tf.float32, [None, 1, max_len], name="ic")
input_q = tf.placeholder(tf.float32, [None, 1, max_len], name="iq")
input_r = tf.placeholder(tf.float32, [None, 1, max_len], name="ir")
input_w = tf.placeholder(tf.float32, [None, 1, max_len], name="iw")
state = tf.placeholder(tf.float32, [None, max_len], name="state")
dropout = tf.placeholder(tf.float32, name="dropout")

zero_input = [np.random.randn(1, max_len) for _ in range(batch_size)]
zero_state = [np.random.randn(max_len) for _ in range(batch_size)]
costs = []

sess = tf.Session()
model = GRUModel(input_c, input_q, input_r, input_w, state, dropout, num_hidden=max_len)
model.load(sess, save_dir='save', dataset=dataset)

# ==================================================
def encode(c_batch, q_batch):
    def merge(article, question):
        prev = zero_state
        for sent in article:
            prev = sess.run(model.prediction, {
                input_c: [sent],
                input_q: [question],
                input_r: zero_input, input_w: zero_input, state: prev, dropout: 0})
            return prev
    assert len(c_batch) == len(q_batch), "Must input same bacth size of context and question"
    encode_batch = [merge(c, q) for c, q in zip(c_batch, q_batch)]
    return encode_batch

# ==================================================
num_correct = 0

print("Running Model...")
for epoch in range(len(test_data)):
# for epoch in range(1):

    idx = np.random.randint(len(test_data))
    # generate batches
    batch_iq = [[iq[idx]]] * batch_size
    batch_ir = [[ir[idx]]] * batch_size
    batch_iw = [[iw[idx]]] * batch_size
    # batch_answers
    answers = chioces[idx]
    batch_ans1 = [[answers[0]]] * batch_size
    batch_ans2 = [[answers[1]]] * batch_size
    batch_ans3 = [[answers[2]]] * batch_size
    batch_ans4 = [[answers[3]]] * batch_size
    # batch_context
    c_batch = [ic_sents[idx]] * batch_size
    c_batch = [[[y] for y in x] for x in c_batch]
    # encode context & question for all context sentences
    batch_enc = encode(c_batch, batch_iq)

    # evaluate on training data
    error = sess.run(model.cosine_cost, {
        input_c: zero_input, input_q: zero_input, input_r: batch_ir, input_w: batch_iw, state: batch_enc[0], dropout: 1})
    costs.append(error)
    # evaluate chioces
    sims = []
    for x in (batch_ans1, batch_ans2, batch_ans3, batch_ans4):
        sims.append(sess.run(model.cosine_cost, {
            input_c: zero_input, input_q: zero_input, input_r: x, input_w: x, state: batch_enc[0], dropout: 1}))
    best_ans = sims.index(min(sims))
    true_ans = truths[idx]
    if best_ans == int(true_ans):
        num_correct += 1
    print('Question {:3d} cosine cost: {:2.5f}, mean cost: {:2.5f}. guess: {:d}, true: {:d}, correct rate: {:3.1f}%'.format(
        epoch + 1, error, sum(costs) / len(costs), best_ans, true_ans, num_correct / (epoch + 1) * 100))
    print('Sims: {:3.5f} {:3.5f} {:3.5f} {:3.5f}'.format(sims[0], sims[1], sims[2], sims[3]))
