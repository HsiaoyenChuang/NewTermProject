import json
import os
import re
import pickle as pkl
import numpy as np
from tensorflow.contrib import learn
import collections

ROOT_PATH = os.getcwd()

DATASET = {
    'squad': 'Squad',
    'qa': 'QA'
}

# Return testing data
def test_data(name='squad'):
    path = '/dataset/%s/testing_data.json' % DATASET[name]
    with open(ROOT_PATH + path) as f:
        data = json.loads(f.read())
    sents = [Sentence(x) for x in data]
    qq = [x.q() for x in sents]
    ll = [x.l() for x in sents]
    cc = [x.c() for x in sents]
    return qq, ll, cc

# Return training data
def train_data(name='squad'):
    path = '/dataset/%s/training_data.json' % DATASET[name]
    with open(ROOT_PATH + path) as f:
        data = json.loads(f.read())
    sents = [Sentence(x) for x in data]
    qq = [x.q() for x in sents]
    ll = [x.l() for x in sents]
    cc = [x.c() for x in sents]
    aa = [x.a() for x in sents]
    return qq, ll, cc, aa

# Transform text sentences to sequence of ids with padding zeros
# def build_vocab(text, processor=None):
#     assert isinstance(text, collections.Iterable), "Must be iterable"
#     if not processor:
#         assert os.path.isfile('save/vocab.pkl'), 'save/vocab.pkl not found!'
#         processor = learn.preprocessing.VocabularyProcessor.restore('save/vocab.pkl')
#     return np.array(list(processor.fit_transform(text)))

def text2vec(sent, vec_dict):
    words = sent.split()
    zero_v = [0] * len(vec_dict['the'])
    sent_vec = []
    for w in words:
        try:
            sent_vec.append(vec_dict[w])
        except KeyError:
            sent_vec.append(zero_v)
    return sent_vec

def normalize_context(s):
    return ' '.join([re.sub(r'[—]', ' ', re.sub(r'[\/\\\']', '', x.strip().lower())) for x in re.split(r'[;\.\(\):,\?\[\]"]', s) if x.strip() != ''])

def normalize(s):
    return re.sub(r'[\/\\\'?—]', ' ', s).strip().lower()

class Sentence(object):
    def __init__(self, raw):
        self._raw = raw
    def c(self):
        return normalize_context(self._raw['context'])
    def q(self):
        return normalize(self._raw['question'])
    def l(self):
        return [normalize(x) for x in self._raw['answer_list']]
    def a(self):
        return self._raw['answer'] if 'answer' in self._raw.keys() else None

def read_glove(words):
    vec = {}
    with open(os.getenv('HOME') + '/data/glove/glove.6B.300d.txt') as f:
        for x in f.readlines():
            vv = x.split()
            if vv[0] in words:
                vec[vv[0]] = [float(i) for i in vv[1:]]
    return vec

def unique_words(lst):
    words = []
    for l in lst:
        for x in l:
            words += x.split()
    return set(words)
