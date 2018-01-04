import line_profiler
import numpy as np

from src import skip_gram

num_words = 100000
sequence_length = 100000
window_size = 5
negative_samples = 5
batch_size = 512
sequence = np.random.randint(low=0, high=num_words, size=sequence_length, dtype=np.int)

profile = line_profiler.LineProfiler(skip_gram.skip_gram_iterator)
profile.runcall(skip_gram.skip_gram_iterator, sequence, window_size, negative_samples)
profile.print_stats()
