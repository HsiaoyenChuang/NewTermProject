import tensorflow as tf
import numpy as np
import preprocess
from model import GRUModel
from tensorflow.contrib import learn

# Data Preparatopn
# ==================================================
# Load data
print("Loading text data...")
qq, ll, cc, aa = preprocess.train_data()
assert len(qq) == len(ll) == len(cc) == len(aa)
# Build vocabulary using same preprocessor
print("Building vocabularies...")
c = [y for x in cc for y in x]
max_len = max([len(x) for x in c])
vocab = learn.preprocessing.VocabularyProcessor(max_len)
vocab.fit(c)
vocab.save('save/vocab.pkl')

print("Convert text to data...")
train_q = preprocess.build_vocab(qq, vocab)
train_c = [preprocess.build_vocab(x, vocab) for x in cc]
train_l = [preprocess.build_vocab(x, vocab) for x in ll]
train_a = [train_l[i][x-1] for i, x in enumerate(aa)]
assert len(train_q) == len(train_c) == len(train_l) == len(train_a)
# Shuffle data
def shuf_data(data):
    ''' Return shuf_data, each consists of:
        - [context, question, right_ans, wrong_ans]
    '''
    print("Shuffle data...")
    shuf_idx = np.random.permutation(np.arange(len(data)))
    data_shuf = [data[i] for i in shuf_idx]
    print("Done shuffle.")
    return data_shuf
def generate_dataset():
    result = []
    for i, c in enumerate(train_c):
        r = train_a[i]
        q = train_q[i]
        for w in train_l[i]:
            if (r == w).all():
                continue
            result.append([c, q, r, w])
    return [[y, x[1], x[2], x[3]] for x in result for y in x[0]]
    # return result
# Training data
train_data = shuf_data(generate_dataset())
assert len(train_data) > 0
assert len(train_data[0]) == 4
print("Max sentence length: {:d}".format(max_len))
print("Vocabulary Size: {:d}".format(len(vocab.vocabulary_)))
print("Train Question Size: {:d}".format(len(train_data)))
# print("Train/Test split:")
ic, iq, ir, iw = zip(*train_data)
assert len(ic) == len(iq) == len(ir) == len(iw) == len(train_data)
# Build Model
# ==================================================
row_size, rows = len(iq[0]), 1
num_classes = len(ir[0])
input_c = tf.placeholder(tf.float32, [None, rows, row_size])
input_q = tf.placeholder(tf.float32, [None, rows, row_size])
input_r = tf.placeholder(tf.float32, [None, rows, row_size])
input_w = tf.placeholder(tf.float32, [None, rows, row_size])
dropout = tf.placeholder(tf.float32)
model = GRUModel(input_c, input_q, input_r, input_w, dropout, num_hidden=max_len)
# Train Model
# ==================================================
print("Running Model...")
sess = tf.Session()
sess.run(tf.initialize_all_variables())
for epoch in range(10):
    for _ in range(100):
        # generate batches
        idx = np.random.randint(len(iq), size=10) # batch_size is 10
        batch_ic = [[ic[i]] for i in idx] # Input iq: batch_size x [1 x max_len]
        batch_iq = [[iq[i]] for i in idx] # Input iq: batch_size x [1 x max_len]
        batch_ir = [[ir[i]] for i in idx] # Input ir: batch_size x [max_len]
        batch_iw = [[iw[i]] for i in idx] # Input ir: batch_size x [max_len]
        # run training procedure
        sess.run(model.optimize, {
            input_c: batch_ic, input_q: batch_iq, input_r: batch_ir, input_w: batch_iw, dropout: 0})
    # evaluate on training data
    error = sess.run(model.evaluate, {
        input_c: batch_ic, input_q: batch_iq, input_r: batch_ir, input_w: batch_iw, dropout: 1})
    print('Epoch {:2d} cosine distance {:3.1f}'.format(epoch + 1, error))
