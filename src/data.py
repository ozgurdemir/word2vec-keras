import numpy as np
from collections import defaultdict
import logging


class Data:
    """ This class is used to hold all data relevant structures such as
    the index and the reverse index
    """

    def __init__(self, file_name):
        self.file_name = file_name
        self.vocab_size = 0
        self.word_occurrence = None
        self.word2Index = None
        self.index2Word = None

    def word_dict(self):
        """ Return a dict of the vocabulary """
        logging.info("Building up word dictionary")
        self.word_occurrence = defaultdict(int)
        with open(self.file_name) as f:
            for line in f:
                for word in line.split("\t"):
                    self.word_occurrence[int(word)] += 1

    def buildIndex(self):
        """ Builds an index and reversed index between words and dense indices.
        The i-ths word relevancy
        """
        logging.info("Building up word indices")
        self.vocab_size = len(self.word_occurrence)
        maxWord = max(self.word_occurrence.keys())
        self.word2Index = np.empty(maxWord + 1, dtype="int64")
        self.index2Word = np.empty(self.vocab_size, dtype="int64")
        sorted_dict = sorted(self.word_occurrence.items(), key=lambda x: x[1], reverse=True)
        for (i, (word, occurrence)) in enumerate(sorted_dict):
            self.index2Word[i] = word
            self.word2Index[word] = i
