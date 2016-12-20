import json
import os
import re
import pickle as pkl
import numpy as np
from tensorflow.contrib import learn

ROOT_PATH = os.getcwd()

# Return testing data
def test_data():
    with open(ROOT_PATH + '/dataset/Squad/testing_data.json') as f:
        data = json.loads(f.read())
    sents = [Sentence(x) for x in data]
    qq = [x.q() for x in sents]
    ll = [x.l() for x in sents]
    cc = [x.c() for x in sents]
    return qq, ll, cc

# Return training data
def train_data():
    with open(ROOT_PATH + '/dataset/Squad/training_data.json') as f:
        data = json.loads(f.read())
    sents = [Sentence(x) for x in data]
    qq = [x.q() for x in sents]
    ll = [x.l() for x in sents]
    cc = [x.c() for x in sents]
    aa = [x.a() for x in sents]
    return qq, ll, cc, aa

# Transform text sentences to sequence of ids with padding zeros
def build_vocab(text, processor=None):
    assert type(text) == list
    if not processor:
        assert os.path.isfile('save/vocab.pkl'), 'save/vocab.pkl not found!'
        processor = learn.preprocessing.VocabularyProcessor.restore('save/vocab.pkl')
    return np.array(list(processor.fit_transform(text)))

class Sentence(object):
    def __init__(self, raw):
        self._raw = raw
    def c(self):
        return [x.strip() for x in re.split(r'[;\.\(\):,\?\[\]]', self._raw['context']) if x.strip() != '']
    def q(self):
        return self._raw['question'].strip()
    def l(self):
        return [x.strip() for x in self._raw['answer_list']]
    def a(self):
        return self._raw['answer'] if 'answer' in self._raw.keys() else None
