import sys, os
from matplotlib import pyplot as plt
import numpy as np
import seaborn
from common_utils import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import list_metrics


class Metrics:
    """
    Encapsulates all metrics for visualizing similarity between lists.
    All methods in this class have identical signature:
        list1, list2 -> real number

    """
    @classmethod
    def overlap(cls, list1: str, list2: str, *args):
        """ Measures word overlap on a scale of 0 to 1.
        """
        overlap, len1, len2 = list_metrics.get_overlap(list1, list2)
        return 2 * len(overlap) / (len1 + len2)

    @classmethod
    def longest_subsequence(cls, list1: str, list2: str, *args):
        """ Measures length of longest subsequence on a scale of 0 to 1.
        """
        len, len1, len2 = list_metrics.longest_subsequence(list1, list2)
        return 2 * len / (len(list1) + len(list2))

    @classmethod
    def political_diff(cls, list1: str, list2: str, *args):
        """ Measures length of longest subsequence on a scale of 0 to 1.
        """
        pos1, pos2, len1, len2 = list_metrics.get_political_diff(list1, list2, *args)
        return 1 - abs(pos1 / len1 - pos2 / len2)


def heatmap(wordlists, metric, *args):
    """
    Constructs a heatmap using the provided similarity metric.
    The heatmap is saved as a .png file.

    Args:
        wordlists (list): list of filenames, each corresponding to a wordlist
        metric ((list1: str, list2: str, *args) -> float): a method from the Metrics class
        args: other arguments for the metric
    """
    num_lists = len(wordlists)
    assert num_lists >= 2, 'need at least 2 wordlists'

    # Compute pairwise similarity
    sim_mat = np.zeros((num_lists, num_lists))
    for i in range(num_lists):
        for j in range(i + 1, num_lists):
            sim_mat[i, j] = metric(wordlists[i], wordlists[j], *args)
        print('Done list {}'.format(i))
    sim_mat += sim_mat.T  # symmetry :)
    sim_mat[np.arange(num_lists), np.arange(num_lists)] = 1.0  # can we assume metrics are on [0, 1] scale?

    # Create heatmap
    labels = [get_publisher_name(file) for file in wordlists]  # labels are publisher names
    plt.figure(figsize=(7, 5))
    plt.yticks(rotation=0)
    fig = seaborn.heatmap(sim_mat, xticklabels=labels, yticklabels=labels)
    fig.get_figure().savefig('heatmap.png', bbox_inches="tight")


def political_ratios(wordlists, model):
    for wordlist in wordlists:
        ratio = list_metrics.get_political_ratio(wordlist, model)
        # TODO: visualize (e.g. barplot)
        print(get_publisher_name(wordlist), ratio)


def clustermap(wordlists, metric):
    raise NotImplementedError


if __name__ == '__main__':
    # GET POLITICAL RATIOS
    prefix = '../citizen-lab-data/chinese-games/dataset2/'
    grouped_dir = '../citizen-lab-data/chinese-games/dataset2-grouped/'

    lst = [grouped_dir + publisher for publisher in os.listdir(grouped_dir)]
    political_ratios(lst, '../training/model_2021-07-28_19:00:00.pt')

    # HEATMAP FOR A SIMILARITY METRIC
    # lst = [grouped_dir + publisher for publisher in os.listdir(grouped_dir)]  # (grouped)

    # lst = ['catcap#cn.actcap.ayc2-277.apk_FILES#assets#sensitive.txt',
    #        'chukong#com.tencent.tmgp.com.cocos2d.fishingfunqq-20008.apk_FILES#assets#config#badwords.txt',
    #        'giant#com.ztgame.armygo-5.apk_FILES#assets#Config#BadWord.txt',
    #        'tencent#com.sgzswljqb.dxkj-1.apk_FILES#assets#res#TXT#badword.txt',
    #        'joymeng#com.joym.fruitattackgarden.yingyongbao-14.apk_FILES#assets#word.txt',
    #        'idreamsky#com.imangi.templerun2-4607.apk_FILES#assets#filter.txt',
    #        'ourpalm#com.tencent.tmgp.zhjol-106.apk_FILES#assets#8c4e80b03c81a26dd53c9953787c1b99.txt',
    #        'xiaoao#com.xiaoao.c5ol.htc-7.apk_FILES#res#raw#pattern.txt',
    #        ]
    # lst = [prefix + file for file in lst] # (not grouped)
    #
    # heatmap(lst, Metrics.political_diff, '../training/model_2021-07-28_19:00:00.pt')
