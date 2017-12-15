import logging

import numpy as np
import pandas as pd
from collections import defaultdict


class Data:
    """ This class contains all data relevant functions
    """

    def __init__(self):
        pass

    @staticmethod
    def read(file_name, field_separator=" "):
        """ Reads a file into memory """
        logging.info("Reading file: %s", file_name)
        sequence = []
        with open(file_name) as f:
            for line in f:
                for word in line.split(field_separator):
                    sequence.append(word.strip())
        logging.info("Number of words %d", len(sequence))
        return sequence

    @staticmethod
    def word_dict(sequence):
        """ Builds up the vocabulary """
        logging.info("Building up word dictionary")
        word_occurrence = defaultdict(int)
        for word in sequence:
            word_occurrence[word] += 1
        logging.info("Vocabulary size %d", len(word_occurrence))
        return word_occurrence

    @staticmethod
    def prune_occurrence(word_occurrence, threshold):
        """ Removes words that occur less than threshold times from the vocabulary """
        word_occurrence = {k: v for k, v in word_occurrence.items() if v >= threshold}
        logging.info("Pruned Vocabulary size: %d", len(word_occurrence))
        return word_occurrence

    @staticmethod
    def word_index(word_occurrence):
        """ Builds an index and reversed index between words and dense indices """
        logging.info("Building up word indices")
        word2index = dict()
        index2word = dict()
        for (i, word) in enumerate(sorted(word_occurrence)):
            index2word[i] = word
            word2index[word] = i
        return word2index, index2word

    @staticmethod
    def re_index(sequence, word2index):
        sequence = [word2index[word] for word in sequence if word in word2index]
        logging.info("Pruned number of words: %d", len(sequence))
        return sequence

    @staticmethod
    def write_embeddings(path, index2word, embeddings):
        """ Saves embeddings to a file """
        logging.info("Saving embeddings to %s", path)
        np_index = np.empty(shape=len(index2word), dtype=object)
        for index, word in index2word.items():
            np_index[index] = word
        df = pd.DataFrame(data=embeddings, index=np_index)
        df.to_csv(path, float_format="%.4f", header=False)
