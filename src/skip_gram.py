import random
import numpy as np
import logging


class SkipGram:
    """ Simple skip gram iterator. The negative samples are sampled uniformly from the input
    sequence.
    """

    def __init__(self):
        pass

    @staticmethod
    def iterator(sequence, window_size, negative_samples):
        """ An iterator which at each step returns a tuple of (word, context, label) """
        for i in xrange(len(sequence)):
            window_start = max(0, i - window_size)
            window_end = min(len(sequence), i + window_size + 1)
            for j in xrange(window_start, window_end):
                if i != j:
                    yield (sequence[i], sequence[j], 1)

            for negative in xrange(negative_samples):
                j = random.randint(0, len(sequence) - 1)
                yield (sequence[i], sequence[j], 0)

    @staticmethod
    def batch_iterator(sequence, window_size, negative_samples, batch_size):
        """ An iterator which returns training instances in batches """
        iterator = SkipGram.iterator(sequence, window_size, negative_samples)
        epoch = 0
        while True:
            words = np.empty(shape=batch_size, dtype=int)
            contexts = np.empty(shape=batch_size, dtype=int)
            labels = np.empty(shape=batch_size, dtype=int)
            for i in xrange(batch_size):
                try:
                    word, context, label = iterator.next()
                except StopIteration:
                    epoch += 1
                    logging.info("iterated %d times over data set", epoch)
                    iterator = SkipGram.iterator(sequence, window_size, negative_samples)
                    word, context, label = iterator.next()
                words[i] = word
                contexts[i] = context
                labels[i] = label
            yield ([words, contexts], labels)
