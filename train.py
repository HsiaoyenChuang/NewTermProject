# =================================================
# Start training a completely new model.
# And save it after Ctrl-C.
# ==================================================
import tensorflow as tf
import numpy as np
import preprocess
from model import GRUModel
from tensorflow.contrib import learn
from tqdm import tqdm

from config import *

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Data Preparatopn
# ==================================================
print("Loading text data...")
qq, ll, cc, aa = preprocess.train_data(dataset)

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
va = [vl[i][x-1] for i, x in enumerate(aa)]

max_len = max([len(x) for x in vc])
set_len = len(vc)
print('= Max len: {:d}'.format(max_len))

del qq, ll, cc, aa, words

print('= Max sentence length: {:d}'.format(max_len))
print('= Train Question Size: {:d}'.format(set_len))



# Build Model
# ==================================================
print("Building Model...")
row_size, ans_len = vec_len, max([len(x) for x in va])
input_c = tf.placeholder(tf.float32, [None, 1, row_size], name="ic") # word of vc
input_q = tf.placeholder(tf.float32, [None, 1, row_size], name="iq") # word of vq
input_r = tf.placeholder(tf.float32, [None, ans_len, row_size], name="ir")
input_w = tf.placeholder(tf.float32, [None, ans_len, row_size], name="iw")
state = tf.placeholder(tf.float32, [None, row_size], name="state")
dropout = tf.placeholder(tf.float32, name="dropout")
model = GRUModel(input_c, input_q, input_r, input_w, state, dropout, num_hidden=vec_len)

# prepare random data for unused placeholders
zero_parah = np.random.randn(1, row_size)
zero_input = np.random.randn(ans_len, row_size)
zero_state = np.random.randn(row_size)

# print('Running Model...')
# sess = tf.Session()
# sess.run(tf.initialize_all_variables())

def create_batch(tensor, batch_size):
    return [tensor] * batch_size

batch_zero_parah = create_batch(zero_parah, batch_size)
batch_zero_input = create_batch(zero_input, batch_size)
batch_zero_state = create_batch(zero_state, batch_size)

# print('Batch generated...')

# sess.run(model.prediction, {
#     input_c: batch_zero_parah,
#     input_q: batch_zero_parah,
#     input_r: batch_zero_input,
#     input_w: batch_zero_input,
#     state: batch_zero_state,
#     dropout: 0
# })
# print('Done.')

def pad_zero(vv, max_len):
    if max_len < len(vv):
        return vv
    vv += [[0] * vec_len] * (max_len - len(vv))
    return vv

va = [pad_zero(x, ans_len) for x in va]
vl = [[pad_zero(y, ans_len) for y in x] for x in vl]


# Train Model
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



print('Running Model...')
print('= Drop Probability: %f' % drop_prob)
print('= Batch Size: %d' % batch_size)
print('= Max Epoch: %d' % max_epoch)

sess = tf.Session()
try: sess.run(tf.global_variables_initializer())
except: sess.run(tf.initialize_all_variables())
print("Initialized. Start training...")


# def shape(lst):
#     try:
#         return len(lst), len(lst[0]), len(lst[0][0])
#     except:
#         return len(lst), len(lst[0])

try:
    costs = []
    for epoch in range(max_epoch):
        print('Epoch {:d}, {:2.3f}%'.format(epoch + 1, epoch * 100. / max_epoch))

        shuf_idx = np.random.permutation(np.arange(set_len))
        vq = [vq[i] for i in shuf_idx]
        vc = [vc[i] for i in shuf_idx]
        va = [va[i] for i in shuf_idx]
        vl = [vl[i] for i in shuf_idx]
        print('Data shuffled, start epoch.')

        assert len(vq) == len(vc) == len(vl) == len(va) == set_len

        for i in tqdm(range(set_len)):
            #print('=> vc[{:d}]: {:3d}, vq[{:d}]: {:2d}, {:2.3f}%'.format(i, len(vc[i]), i, len(vq[i]), i * 100. / set_len))
            batch_cq = encode(vc[i], vq[i])
            batch_ir = create_batch(va[i], batch_size)
            for opt in vl[i]:
                if opt != va[i]:
                    batch_iw = create_batch(opt, batch_size)
                    # print('ic:', shape(batch_zero_parah), input_c.get_shape())
                    # print('iq:', shape(batch_zero_parah), input_q.get_shape())
                    # print('ir:', shape(batch_ir), input_r.get_shape())
                    # print('iw:', shape(batch_iw), input_w.get_shape())
                    # print('state:', shape(batch_cq), state.get_shape())
                    # raise
                    sess.run(model.optimize, {
                        input_c: batch_zero_parah,
                        input_q: batch_zero_parah,
                        input_r: batch_ir,
                        input_w: batch_iw,
                        state:   batch_cq,
                        dropout: drop_prob
                    })
            # evaluate on training data
            if (i + 1) % 100 == 0:
                error = sess.run(model.cosine_cost, {
                    input_c: batch_zero_parah,
                    input_q: batch_zero_parah,
                    input_r: batch_ir,
                    input_w: batch_iw,
                    state: batch_cq,
                    dropout: 1
                })
                costs.append(error)
                print('=> cosine cost {:3.5f}, mean cost: {:3.5f}'.format(error, sum(costs) / len(costs)))
            # autosave
            if (i + 1) % 1000 == 0:
                model.save(sess, save_dir='save', dataset=dataset)

except KeyboardInterrupt:
    pass

# ensure model is saved.
model.save(sess, save_dir='save', dataset=dataset)

print('Done.')
