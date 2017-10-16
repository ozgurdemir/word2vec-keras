import logging
from collections import defaultdict

import numpy as np
from keras.preprocessing import sequence


class Data:
    """ This class is used to hold all data relevant structures such as
    the index and the reverse index
    """

    field_separator = " "

    def __init__(self, file_name):
        self.file_name = file_name
        self.vocab_size = 0
        self.size = 0
        self.word_occurrence = dict()
        self.word2Index = None
        self.index2Word = None

    def word_dict(self):
        """ Builds a dict of the vocabulary """
        logging.info("Building up word dictionary")
        self.word_occurrence = defaultdict(int)
        with open(self.file_name) as f:
            for line in f:
                for word in line.split(self.field_separator):
                    self.word_occurrence[word.strip()] += 1
                    self.size += 1
        logging.info("Found %d words and %d unique words", self.size, len(self.word_occurrence))

    def word_index(self):
        """ Builds an index and reversed index between words and dense indices.
        The index of a word is its rank in the data set (starting at 1).
        """
        logging.info("Building up word indices")
        self.vocab_size = len(self.word_occurrence)
        self.word2Index = dict()
        self.index2Word = dict()
        sorted_dict = sorted(self.word_occurrence.items(), key=lambda x: x[1], reverse=True)
        for (i, (word, occurrence)) in enumerate(sorted_dict):
            word = word.strip()
            self.index2Word[i] = word
            self.word2Index[word] = i

    def skip_gram_iterator(self, window_size, negative_samples, shuffle, sampling_table, chunk_size=100):
        """ Returns an iterator over pairs of words either positive or negative with their appropriate label. """
        epoch = 0
        word_target = np.ndarray(shape=(1), dtype=np.int32)
        word_context = np.ndarray(shape=(1), dtype=np.int32)
        label = np.ndarray(shape=(1), dtype=np.int32)
        while True:
            with open(self.file_name) as f:
                for line in f:
                    re_indexed_sentence = [self.word2Index[word.strip()] for word in line.split(self.field_separator)]
                    for chunk in self.chunks(re_indexed_sentence, chunk_size):
                        couples, labels = sequence.skipgrams(chunk, self.vocab_size,
                                                             window_size=window_size,
                                                             negative_samples=negative_samples,
                                                             shuffle=shuffle,
                                                             sampling_table=sampling_table)
                        for (index, couple) in enumerate(couples, start=0):
                            x, y = couple
                            word_target[0] = x
                            word_context[0] = y
                            label[0] = labels[index]
                            yield ([word_target, word_context], label)
            epoch += 1
            logging.info("Iterated %d times over train set", epoch)

    @staticmethod
    def chunks(l, n):
        """Yield successive n-sized chunks from l.
        This is used for input files that contain all text as a single line.
        """
        for i in xrange(0, len(l), n):
            yield l[i:i + n]
