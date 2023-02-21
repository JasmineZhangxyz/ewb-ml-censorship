import sys
import unittest
from common_utils import *

# TODO: ugly. is there a better way to do imports?
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import list_metrics


class MetricsUnitTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.en_files = ['../tests/en_list1.txt',
                        '../tests/en_list2.txt',
                        '../tests/en_ans.txt']
        cls.zh_files = ['../tests/ch_list1.txt',
                        '../tests/ch_list2.txt',
                        '../tests/ch_ans.txt']
        cls.en_lengths = [20, 30, 14]
        cls.zh_lengths = [20, 34, 13]

    def test_overlap_en(self):
        """
        Unit test for the get_overlap() method with English word lists.
        """
        overlap, len1, len2 = list_metrics.get_overlap(
            self.en_files[0], self.en_files[1])

        len1_ans = self.en_lengths[0]
        len2_ans = self.en_lengths[1]
        count_ans = self.en_lengths[2]

        self.assertEqual(len(overlap), count_ans,
                         'Incorrect overlap count, got {}, expected {}'.format(len(overlap), count_ans))
        self.assertEqual(len1, len1_ans,
                         'Incorrect list1 length, got {}, expected {}'.format(len1, len1_ans))
        self.assertEqual(len2, len2_ans,
                         'Incorrect list2 length, got {}, expected {}'.format(len2, len2_ans))

        self.assertListEqual(sorted(overlap), sorted(file_to_list(self.en_files[2])))

    def test_overlap_zh(self):
        """
        Unit test for the get_overlap() method with Chinese word lists.
        """
        overlap, len1, len2 = list_metrics.get_overlap(
            self.zh_files[0], self.zh_files[1])

        len1_ans = self.zh_lengths[0]
        len2_ans = self.zh_lengths[1]
        count_ans = self.zh_lengths[2]
        self.assertEqual(len(overlap), count_ans,
                         'Incorrect overlap count, got {}, expected {}'.format(len(overlap), count_ans))
        self.assertEqual(len1, len1_ans,
                         'Incorrect list1 length, got {}, expected {}'.format(len1, len1_ans))
        self.assertEqual(len2, len2_ans,
                         'Incorrect list2 length, got {}, expected {}'.format(len2, len2_ans))

    def test_longest_subsequence(self):
        list1 = [1, 2, 3, 4, 5, 6]
        list2 = [1, 3, 5]
        self.assertEqual(list_metrics.longest_subsequence(list1, list2), 1)
        self.assertEqual(list_metrics.longest_subsequence(list1, list1), 6)

        list1 = ['m', 'l', 'f', 's']
        list2 = ['c', 'e', 'd', 'r', 'm', 'l', 'f', 's', '!']
        self.assertEqual(list_metrics.longest_subsequence(list1, list2), 4)
        self.assertEqual(list_metrics.longest_subsequence(list1, list2[0:4]), 0)

        list1 = ['奶牛', '兔子', '鸭子', '鹦鹉']
        list2 = ['虾]', '兔子', '鸭子']
        self.assertEqual(list_metrics.longest_subsequence(list1, list2), 2)


if __name__ == '__main__':
    unittest.main()
