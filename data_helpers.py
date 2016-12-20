import numpy as np

def _l2_norm(x, axis):
    norm = np.sqrt(np.sum(np.square(x), axis=axis, keepdims=True))
    return x / norm

def _cos_sim(x, y):
    '''Cosine Similarity between x and y,
        - x: truth, y: guess
    '''
    assert x.ndim == y.ndim == 2
    return np.sum(x * y, axis=1, keepdims=False)
