import numpy as np
import torch

from snownlp import SnowNLP
from common_utils import *
from preprocessing.clean_data import batchify


def get_overlap(list1, list2):
    """
    Returns a list of words that occur in both list1 and list2.
    Also returns total number of words in list1 and in list2 (can be used
    to compute similarity as a percentage, if desired)

    Args:
        list1 (string): filename corresponding to a list of words
        list2 (string): filename corresponding to a list of words

    Returns:
        overlap (list of string): words occuring in both list1 and list2
        len1 (int): number of words in list1
        len2 (int): number of words in list2
    """
    # File to List
    wordlist1 = file_to_str(list1)
    wordlist2 = file_to_str(list2)

    # Tokenize Lists
    s1 = SnowNLP(wordlist1)  # need to use SnowNLP for Chinese-character lists
    s2 = SnowNLP(wordlist2)

    # Get List Lengths
    wordlist1 = list(set(s1.words))
    wordlist2 = list(set(s2.words))
    len1 = len(wordlist1)
    len2 = len(wordlist2)

    # Count Overlapping Words
    # overlap = [w2 for w2 in wordlist2 if w2 in wordlist1]

    # Alternative to "Count Overlapping Words" with potentially better time complexity
    combined_wordset = set(wordlist1).union(set(wordlist2))
    overlap = len1 + len2 - len(combined_wordset)

    return overlap, len1, len2


def get_embedding_similarity(list1, list2, emb):
    """
    Given an embedding for a vocabulary and two lists of words from
    that vocabulary, return the cosine distance between the average
    word embeddings from each list.

    Args:
        list1 (string): filename corresponding to a list of words
        list2 (string): filename corresponding to a list of words
        emb (??): word embedding
    """
    raise NotImplementedError


def get_political_diff(list1, list2, model_file):
    # Load model and data
    model = torch.load(model_file)
    model.eval()
    data1 = batchify(list1)
    data2 = batchify(list2)

    # Inference
    positives1, positives2 = 0, 0
    for in1 in data1:
        output1 = model(in1[0])  # no labels
        positives1 += (output1.argmax(1) == 1).sum().item()
    for in2 in data2:
        output2 = model(in2[0])
        positives2 += (output2.argmax(1) == 1).sum().item() / len(data2)
    return positives1, positives2, len(data1.dataset), len(data2.dataset)


def get_political_ratio(list, model_file):
    # Load model and data
    model = torch.load(model_file)
    model.eval()
    data = batchify(list)

    # Inference
    positives = 0
    for in1 in data:
        output = model(in1[0])  # no labels
        positives += (output.argmax(1) == 1).sum().item()
    return positives / len(data.dataset)


def get_longest_subsequence_length(list1, list2):
    """
    Returns length of longest subsequence common to list1 and list2.
    Also returns total number of words in list1 and in list2 (can be used
    to compute similarity as a percentage, if desired)

    Args:
        list1 (string): filename corresponding to a list of words
        list2 (string): filename corresponding to a list of words

    Returns:
        length (int): length of longest subsequence
        len1 (int): number of words in list1
        len2 (int): number of words in list2
    """
    # File to List
    wordlist1 = file_to_str(list1)
    wordlist2 = file_to_str(list2)

    # Tokenize Lists
    s1 = SnowNLP(wordlist1)
    s2 = SnowNLP(wordlist2)

    # Get List Lengths
    wordlist1 = list(s1.words)
    wordlist2 = list(s2.words)
    len1 = len(wordlist1)
    len2 = len(wordlist2)

    length = longest_subsequence(wordlist1, wordlist2)

    return length, len1, len2


def longest_subsequence(list1, list2):
    """
    Return length of longest subsequence common to list1 and list2.

    Here, a subsequence is defined as a list of ordered entries
    occuring consecutively in a list.

    Example:
        [1,2,3] and [4] are subsequences of [1,2,3,4]
        [1,3,4] and [1,3,2] are not subsequences of [1,2,3,4]

    Args:
        list1 (list): a list whose elements can be of any class
            that implements __eq__
        list2 (list): a list whose elements are the same class as
            those of list1

    Returns:
        l (int): the length of the longest subsequence
    """
    # T[i, j] will store the length of longest substring
    # ending at list1[i] and list2[j]
    n = len(list1)
    m = len(list2)
    T = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            if list1[i] == list2[j]:
                if i == 0 or j == 0:
                    T[i, j] = 1
                else:
                    T[i, j] = T[i-1, j-1] + 1

    return np.max(T)


def _cosine_dist(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def _euclidean_dist(v1, v2):
    return np.linalg.norm(v1 - v2)


def _manhattan_dist(v1, v2):
    raise np.linalg.norm(v1 - v2, ord=1)
