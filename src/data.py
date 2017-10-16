import logging
from collections import defaultdict

import numpy as np
from keras.preprocessing import sequence


class Data:
    """ This class is used to hold all data relevant structures such as
    the index and the reverse index
    """

    field_separator = "\t"

    def __init__(self, file_name):
        self.file_name = file_name
        self.vocab_size = 0
        self.size = 0
        self.word_occurrence = None
        self.word2Index = None
        self.index2Word = None

    def word_dict(self):
        """ Builds a dict of the vocabulary """
        logging.info("Building up word dictionary")
        self.word_occurrence = defaultdict(int)
        with open(self.file_name) as f:
            for line in f:
                for word in line.split(self.field_separator):
                    self.word_occurrence[int(word)] += 1
                    self.size += 1
        logging.info("Found %d words and %d unique words", self.size, len(self.word_occurrence))

    def word_index(self):
        """ Builds an index and reversed index between words and dense indices.
        The index of a word is its rank in the data set (starting at 1).
        """
        logging.info("Building up word indices")
        self.vocab_size = len(self.word_occurrence)
        max_word = max(self.word_occurrence.keys())
        self.word2Index = np.empty(max_word + 1, dtype="int64")
        self.index2Word = np.empty(self.vocab_size, dtype="int64")
        sorted_dict = sorted(self.word_occurrence.items(), key=lambda x: x[1], reverse=True)
        for (i, (word, occurrence)) in enumerate(sorted_dict):
            self.index2Word[i] = word
            self.word2Index[word] = i

    def skip_gram_iterator(self, window_size, negative_samples, shuffle, sampling_table):
        """ Returns an iterator over pairs of words either positive or negative with their appropriate label. """
        epoch = 0
        while True:
            with open(self.file_name) as f:
                line_number = 0
                for line in f:
                    line_number += 1
                    if line_number % 1000 == 0:
                        logging.info("Reading line: %d", line_number)
                    re_indexed_sentence = [self.word2Index[int(word)] for word in line.split("\t")]
                    couples, labels = sequence.skipgrams(re_indexed_sentence, self.vocab_size,
                                                         window_size=window_size,
                                                         negative_samples=negative_samples,
                                                         shuffle=shuffle,
                                                         sampling_table=sampling_table)
                    for (index, couple) in enumerate(couples, start=0):
                        x, y = couple
                        word_target = np.array([x], dtype="int32")
                        word_context = np.array([y], dtype="int32")
                        label = np.array([labels[index]], dtype="int32")
                        yield ([word_target, word_context], label)
            epoch += 1
            logging.info("Iterated %d times over train set", epoch)
