import unittest

import numpy as np
from keras.preprocessing import sequence

from data import Data


class TestData(unittest.TestCase):
    def setUp(self):
        train_fixture = "fixtures/train.txt"
        self.data = Data(train_fixture)
        self.data.word_dict()
        self.data.word_index()

    def test_size(self):
        self.assertEqual(self.data.size, 12)

    def test_word_dict(self):
        word_dict = {1: 2, 2: 2, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1}
        self.assertEqual(self.data.word_occurrence, word_dict)

    def test_vocab_size(self):
        self.assertEqual(self.data.vocab_size, 8)

    def test_word2index(self):
        word2index = np.arange(0, 8)
        np.testing.assert_array_equal(self.data.word2Index[1:], word2index)

    def test_index2word(self):
        index2word = np.arange(1, 9)
        np.testing.assert_array_equal(self.data.index2Word, index2word)

    def test_iterator(self):
        sampling_table = sequence.make_sampling_table(self.data.vocab_size, 1)
        iterator = self.data.skip_gram_iterator(window_size=1, negative_samples=1, shuffle=False,
                                                sampling_table=sampling_table)
        self.assertEqual(iterator.next(), ([[1], [2]], 1))
        # self.assertEqual(iterator.next(), ([2, 1], 1))
        # self.assertEqual(iterator.next(), ([2, 3], 1))
        # self.assertEqual(iterator.next(), ([3, 2], 1))


if __name__ == '__main__':
    unittest.main()
