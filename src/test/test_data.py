import io
import os
import unittest

import numpy as np

import data


class TestData(unittest.TestCase):
    def test_read(self):
        train_fixture = os.path.join(os.path.dirname(__file__), "fixtures/train.txt")
        got = data.read(train_fixture)
        want = ['1', '2', '3', '4', '5', '6', '7', '8', '1', '2', '3', '4']
        self.assertEqual(got, want)

    def test_word_dict(self):
        sequence = ['1', '2', '3', '4', '5', '6', '7', '8', '1', '2', '3', '4']
        got = data.word_dict(sequence)
        want = {'1': 2, '3': 2, '2': 2, '5': 1, '4': 2, '7': 1, '6': 1, '8': 1}
        self.assertEqual(got, want)

    def test_prune_occurrence(self):
        word_occurrence = {'1': 2, '3': 2, '2': 2, '5': 1, '4': 2, '7': 1, '6': 1, '8': 1}
        got = data.prune_occurrence(word_occurrence, 2)
        want = {'1': 2, '3': 2, '2': 2, '4': 2}
        self.assertEqual(got, want)

    def test_word_index(self):
        word_occurrence = {'1': 2, '2': 2, '3': 2}
        got = data.word_index(word_occurrence)
        want = ({'1': 0, '2': 1, '3': 2}, {0: '1', 1: '2', 2: '3'})
        self.assertEqual(got, want)

    def test_re_index(self):
        sequence = ['1', '2', '3']
        word2index = {'1': 0, '2': 1, '3': 2}
        got = data.re_index(sequence, word2index)
        want = [0, 1, 2]
        self.assertEqual(got, want)

    def test_write_embeddings(self):
        embeddings = np.array([[1, 2, 3], [4, 5, 6]])
        index2word = {1: "second", 0: "first"}
        buffer = io.StringIO()
        data.write_embeddings(buffer, index2word, embeddings)
        wanted = "first,1,2,3\nsecond,4,5,6\n"
        self.assertEqual(buffer.getvalue(), wanted)


if __name__ == '__main__':
    unittest.main()
