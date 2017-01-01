# ==================================================
# Load and test a trained model on Traing Set.
# ==================================================
import tensorflow as tf
import numpy as np
import preprocess
from model import GRUModel
from tensorflow.contrib import learn

from config import *

print("Loading text data...")
qq, ll, cc = preprocess.test_data(dataset)

words = preprocess.unique_words([cc, qq])
print('= Unique words: {:d}'.format(len(words)))

vec = preprocess.read_glove(words) # {key: word_vec}
print('= Found vectors: {:d}'.format(len(vec)))

vec_len = len(vec['the'])
print('= Vec len: {:d}'.format(vec_len))

print("Text to vec...")
vq = [preprocess.text2vec(x, vec) for x in qq] # [[vec, ..], [vec, ...], ...], each [vec, ...] is a sentence
vc = [preprocess.text2vec(x, vec) for x in cc] # [[vec, ..], [vec, ...], ...], each [vec, ...] is a paragraph
vl = [[preprocess.text2vec(y, vec) for y in x] for x in ll] # [[vec, ..], [vec, ...], [vec, ...], [vec, ...]]

max_len = max([len(x) for x in vc])
set_len = len(vc)
print('= Max len: {:d}'.format(max_len))

del qq, ll, cc, words

print('= Max sentence length: {:d}'.format(max_len))
print('= Train Question Size: {:d}'.format(set_len))

# ==================================================
print("Rstoring Model...")
va = [y for x in vl for y in x]
row_size, ans_len = vec_len, max([len(x) for x in va])
input_c = tf.placeholder(tf.float32, [None, 1, row_size], name="ic") # word of vc
input_q = tf.placeholder(tf.float32, [None, 1, row_size], name="iq") # word of vq
input_r = tf.placeholder(tf.float32, [None, ans_len, row_size], name="ir")
input_w = tf.placeholder(tf.float32, [None, ans_len, row_size], name="iw")
state = tf.placeholder(tf.float32, [None, row_size], name="state")
dropout = tf.placeholder(tf.float32, name="dropout")

zero_parah = np.random.randn(1, row_size)
zero_input = np.random.randn(ans_len, row_size)
zero_state = np.random.randn(row_size)

def create_batch(tensor, batch_size):
    return [tensor] * batch_size

batch_zero_parah = create_batch(zero_parah, batch_size)
batch_zero_input = create_batch(zero_input, batch_size)
batch_zero_state = create_batch(zero_state, batch_size)

costs = []

sess = tf.Session()
model = GRUModel(input_c, input_q, input_r, input_w, state, dropout, num_hidden=vec_len)
model.load(sess, save_dir='save', dataset=dataset)

# ==================================================
def encode(v, q):
    prev = batch_zero_state
    for x in q:
        batch_q = create_batch([x], batch_size) # each word from vq
        for y in v:
            batch_w = create_batch([y], batch_size) # each word from vc
            prev = sess.run(model.prediction, {
                input_c: batch_w,
                input_q: batch_q,
                input_r: batch_zero_input,
                input_w: batch_zero_input,
                state: batch_zero_state,
                dropout: 0
            })
    return prev

def pad_zero(vv, max_len):
    if max_len < len(vv):
        return vv
    vv += [[0] * vec_len] * (max_len - len(vv))
    return vv

vl = [[pad_zero(y, ans_len) for y in x] for x in vl]


# ==================================================
print('Running Model...')
print('= Drop Probability: %f' % drop_prob)
print('= Batch Size: %d' % batch_size)
print('= Max Epoch: %d' % max_epoch)

# f = open('answer.txt', 'w+')
num_correct = 0

# costs = []
for i in range(set_len):
    sims = []
    batch_cq = encode(vc[i], vq[i])

    for va in vl[i]:
        batch_ir = create_batch(va, batch_size)
        # evaluate on training data
        # error = sess.run(model.cosine_cost, {
        #     input_c: batch_zero_parah,
        #     input_q: batch_zero_parah,
        #     input_r: batch_ir,
        #     input_w: batch_ir,
        #     state: batch_cq,
        #     dropout: 1
        # })
        sims.append(sess.run(model.evaluate, {
            input_c: batch_zero_parah,
            input_q: batch_zero_parah,
            input_r: batch_ir,
            input_w: batch_ir,
            state: batch_cq,
            dropout: 1
        }))
        # costs.append(error)
    # print('=> cosine cost {:3.5f}, mean cost: {:3.5f}'.format(error, sum(costs) / len(costs)))
    print('Question {:d}, sims:'.format(i+1))
    print(sims)


# for epoch in range(len(test_data)):
#     idx = epoch
#     # generate batches
#     batch_iq = [[iq[idx]]] * batch_size
#     # batch_answers
#     answers = chioces[idx]
#     batch_ans1 = [[answers[0]]] * batch_size
#     batch_ans2 = [[answers[1]]] * batch_size
#     batch_ans3 = [[answers[2]]] * batch_size
#     batch_ans4 = [[answers[3]]] * batch_size
#     # batch_context
#     c_batch = [ic_sents[idx]] * batch_size
#     c_batch = [[[y] for y in x] for x in c_batch]
#     # encode context & question for all context sentences
#     batch_enc = encode(c_batch, batch_iq)

#     sims = []
#     for x in (batch_ans1, batch_ans2, batch_ans3, batch_ans4):
#         sims.append(sess.run(model.cosine_cost, {
#             input_c: zero_input, input_q: zero_input, input_r: x, input_w: x, state: batch_enc[0], dropout: 1}))
#     best_ans = sims.index(min(sims))
#     f.write("%d\n" % best_ans)

# f.close()
