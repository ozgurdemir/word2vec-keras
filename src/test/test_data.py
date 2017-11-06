import unittest

from data import Data


class TestData(unittest.TestCase):
    def test_read(self):
        train_fixture = "src/test/fixtures/train.txt"
        got = Data.read(train_fixture)
        want = ['1', '2', '3', '4', '5', '6', '7', '8', '1', '2', '3', '4']
        self.assertEqual(got, want)

    def test_word_dict(self):
        sequence = ['1', '2', '3', '4', '5', '6', '7', '8', '1', '2', '3', '4']
        got = Data.word_dict(sequence)
        want = {'1': 2, '3': 2, '2': 2, '5': 1, '4': 2, '7': 1, '6': 1, '8': 1}
        self.assertEqual(got, want)

    def test_prune_occurrence(self):
        word_occurrence = {'1': 2, '3': 2, '2': 2, '5': 1, '4': 2, '7': 1, '6': 1, '8': 1}
        got = Data.prune_occurrence(word_occurrence, 2)
        want = {'1': 2, '3': 2, '2': 2, '4': 2}
        self.assertEqual(got, want)

    def test_word_index(self):
        word_occurrence = {'1': 2, '2': 2, '3': 2}
        got = Data.word_index(word_occurrence)
        want = ({'1': 0, '2': 2, '3': 1}, {0: '1', 1: '3', 2: '2'})
        self.assertEqual(got, want)

    def test_re_index(self):
        sequence = ['1', '2', '3']
        word2index = {'1': 0, '2': 1, '3': 2}
        got = Data.re_index(sequence, word2index)
        want = [0, 1, 2]
        self.assertEqual(got, want)


if __name__ == '__main__':
    unittest.main()
