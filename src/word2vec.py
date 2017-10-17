import logging

import numpy as np
from keras.layers import Input, Dense
from keras.layers.core import Flatten
from keras.layers.embeddings import Embedding
from keras.layers.merge import dot
from keras.models import Model
from keras.optimizers import Adagrad
from keras.utils import plot_model

from skip_gram import SkipGram


class Word2Vec:
    """ A word2vec model implemented in keras. This is an implementation that
    uses negative sampling and skipgrams. """

    def __init__(self):
        self.model = None

    def build(self, vector_dim, vocab_size):
        """ returns a word2vec model """
        logging.info("Building keras model")
        word_input = Input(shape=(1,), name="word_input")
        word = Embedding(input_dim=vocab_size, output_dim=vector_dim, input_length=1, name="word_embedding")(
            word_input)

        context_input = Input(shape=(1,), name="context_input")
        context = Embedding(input_dim=vocab_size, output_dim=vector_dim, input_length=1,
                            name="context_embedding")(
            context_input)

        merged = dot([word, context], axes=2, normalize=False, name="dot")
        merged = Flatten()(merged)
        output = Dense(1, activation='sigmoid', name="output")(merged)

        optimizer = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
        model = Model(inputs=[word_input, context_input], outputs=output)
        model.compile(loss="binary_crossentropy", optimizer=optimizer)
        self.model = model

    def train(self, sequence, window_size, negative_samples, batch_size, epochs, workers, verbose):
        """ Trains the word2vec model """
        logging.info("Training model")

        # in order to balance out more negative samples than positive
        negative_weight = 1.0 / negative_samples
        class_weight = {1: 1.0, 0: negative_weight}
        logging.info("Class weights set to: %s", class_weight)

        num_words = long(len(sequence))
        approx_steps_per_epoch = (num_words * (window_size * 2.0) + num_words * negative_samples) / batch_size
        logging.info("Approx. steps per epoch: %d", approx_steps_per_epoch)
        skip_gram_iterator = SkipGram.batch_iterator(sequence, window_size, negative_samples, batch_size)

        self.model.fit_generator(skip_gram_iterator,
                                 steps_per_epoch=approx_steps_per_epoch,
                                 epochs=epochs,
                                 verbose=verbose,
                                 class_weight=class_weight,
                                 max_queue_size=100,
                                 workers=workers)

    def write_embeddings(self, path, index2word):
        """ Saves the embeddings of a file """
        logging.info("Saving embeddings to %s", path)
        with open(path, 'wb') as fout:
            for i in xrange(len(index2word)):
                word = index2word[i]
                weights = self.model.get_layer("word_embedding").get_weights()[0][i]
                fout.write('%d\t' % word)
                fout.write(np.array2string(weights, precision=4, separator=',', prefix="", max_line_width=10000))
                fout.write("\n")
            fout.flush()

    def plot(self, path):
        """ Plots the model to a file"""
        logging.info("Saving model image to: %s", path)
        plot_model(self.model, show_shapes=True, to_file=path)
