# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

import logging
import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.int
ctypedef np.int_t DTYPE_t

from libc.stdlib cimport rand, srand, RAND_MAX

@cython.binding(True)
@cython.boundscheck(False)
@cython.cdivision(True)
def skip_gram_iterator(DTYPE_t[:] sequence, int window_size, int negative_samples, int seed):
    """ An iterator which at each step returns a tuple of (word, context, label) """
    srand(seed);
    cdef int sequence_length = sequence.shape[0]
    cdef int i, j, window_start, window_end, epoch
    cdef float random_float
    epoch = 0
    i = 0
    while True:
        window_start = max(0, i - window_size)
        window_end = min(sequence_length, i + window_size + 1)
        for j in range(window_start, window_end):
            if i != j:
                yield (sequence[i], sequence[j], 1)

        for negative in range(negative_samples):
            random_float = rand() / (RAND_MAX * 1.0)
            j = int(random_float * sequence_length)
            yield (sequence[i], sequence[j], 0)

        i += 1
        if i == sequence_length:
          epoch += 1
          logging.info("iterated %d times over data set", epoch)
          i = 0

@cython.binding(True)
@cython.boundscheck(False)
def batch_iterator(DTYPE_t[:] sequence, int window_size, int negative_samples, int batch_size, int seed):
    """ An iterator which returns training instances in batches """
    iterator = skip_gram_iterator(sequence, window_size, negative_samples, seed)
    cdef DTYPE_t[:] words = np.empty(shape=batch_size, dtype=DTYPE)
    cdef DTYPE_t[:] contexts = np.empty(shape=batch_size, dtype=DTYPE)
    cdef DTYPE_t[:] labels = np.empty(shape=batch_size, dtype=DTYPE)
    cdef int word, context, label
    while True:
        for i in range(batch_size):
          word, context, label = next(iterator)
          words[i] = word
          contexts[i] = context
          labels[i] = label
        yield ([words, contexts], labels)
