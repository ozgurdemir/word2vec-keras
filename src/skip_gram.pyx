# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

import logging
import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.int
ctypedef np.int_t DTYPE_t

from libc.stdlib cimport rand, RAND_MAX

@cython.binding(True)
def skip_gram_iterator(np.ndarray sequence, int window_size, int negative_samples):
    """ An iterator which at each step returns a tuple of (word, context, label) """
    cdef int sequence_length = sequence.shape[0]
    cdef int i, j, window_start, window_end
    for i in range(sequence_length):
        window_start = max(0, i - window_size)
        window_end = min(sequence_length, i + window_size + 1)
        for j in range(window_start, window_end):
            if i != j:
                yield (sequence[i], sequence[j], 1)

        for negative in range(negative_samples):
            j = 1 + int(rand()/(RAND_MAX * sequence_length * 1.0))
            yield (sequence[i], sequence[j], 0)

@cython.binding(True)
def batch_iterator(np.ndarray sequence, int window_size, int negative_samples, int batch_size):
    """ An iterator which returns training instances in batches """
    iterator = skip_gram_iterator(sequence, window_size, negative_samples)
    cdef int epoch = 0
    while True:
        words = np.empty(shape=batch_size, dtype=DTYPE)
        contexts = np.empty(shape=batch_size, dtype=DTYPE)
        labels = np.empty(shape=batch_size, dtype=DTYPE)
        for i in range(batch_size):
            try:
                word, context, label = next(iterator)
            except StopIteration:
                epoch += 1
                logging.info("iterated %d times over data set", epoch)
                iterator = skip_gram_iterator(sequence, window_size, negative_samples)
                word, context, label = next(iterator)
            words[i] = word
            contexts[i] = context
            labels[i] = label
        yield ([words, contexts], labels)
