import unittest

import numpy as np

from src import skip_gram


class TestData(unittest.TestCase):

    def test_skip_gram_iterator(self):
        sequence = np.array([1, 2, 3, 4, 5])
        window_size = 2
        negative_samples = 2
        seed = 1
        skip_gram_iterator = skip_gram.skip_gram_iterator(sequence, window_size, negative_samples, seed)

        self.assertEqual(next(skip_gram_iterator), (1, 2, 1))
        self.assertEqual(next(skip_gram_iterator), (1, 3, 1))
        self.assertEqual(next(skip_gram_iterator), (1, 5, 0))
        self.assertEqual(next(skip_gram_iterator), (1, 2, 0))
        self.assertEqual(next(skip_gram_iterator), (2, 1, 1))
        self.assertEqual(next(skip_gram_iterator), (2, 3, 1))
        self.assertEqual(next(skip_gram_iterator), (2, 4, 1))
        self.assertEqual(next(skip_gram_iterator), (2, 4, 0))
        self.assertEqual(next(skip_gram_iterator), (2, 4, 0))