'''Contains functions that take two Strings instances and return a number
between 0.0 and 1.0, inclusive, based on how similar the instances are to each
other, 1.0 being most similar.
'''

import math

def jaccard(xs, ys):
    '''Returns jaccard similarity between xs and ys.'''
    if not xs and not ys:
        return 0.0 # typically defined as 1.0
    intersection = len(xs.set & ys.set)
    union = len(xs.set | ys.set)
    return intersection / union

def overlap(xs, ys):
    '''Returns the overlap coefficient between xs and ys.'''
    if not xs or not ys:
        return 0.0
    intersection = len(xs.set & ys.set)
    minimum = min(len(xs.set), len(ys.set))
    return intersection / minimum

def euclidean(xs, ys):
    common_words = xs.set | ys.set
    xs_norm = math.sqrt(sum(v * v for v in xs.counter.values()))
    ys_norm = math.sqrt(sum(v * v for v in ys.counter.values()))
    squared_dist = 0
    for word in common_words:
        diff = xs.counter[word] / xs_norm - ys.counter[word] / ys_norm
        squared_dist += diff * diff
    return 1.0 - math.sqrt(squared_dist / 2.0)

def euclidean_set(xs, ys):
    common_words = xs.set | ys.set
    xs_norm = math.sqrt(len(xs.set))
    ys_norm = math.sqrt(len(ys.set))
    squared_dist = 0
    for word in common_words:
        diff = (word in xs.set) / xs_norm - (word in ys.set) / ys_norm
        squared_dist += diff * diff
    return 1.0 - math.sqrt(squared_dist / 2.0)

def cosine(xs, ys):
    common_words = xs.set | ys.set
    xs_norm = math.sqrt(sum(v * v for v in xs.counter.values()))
    ys_norm = math.sqrt(sum(v * v for v in ys.counter.values()))
    dot = 0
    for word in common_words:
        dot += xs.counter[word] * ys.counter[word]
    return dot / (xs_norm * ys_norm)

def cosine_set(xs, ys):
    common_words = xs.set | ys.set
    xs_norm = math.sqrt(len(xs.set))
    ys_norm = math.sqrt(len(ys.set))
    dot = 0
    for word in common_words:
        dot += word in xs.set and word in ys.set
    return dot / (xs_norm * ys_norm)
