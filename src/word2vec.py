import logging

import numpy as np
from keras.layers import Input, Dense
from keras.layers.core import Flatten
from keras.layers.embeddings import Embedding
from keras.layers.merge import dot
from keras.models import Model
from keras.optimizers import Adagrad
from keras.preprocessing import sequence
from keras.utils import plot_model


class Word2Vec:
    """ A word2vec model implemented in keras. This is an implementation that
    uses negative sampling and skipgrams. """

    def __init__(self, data):
        self.data = data
        self.model = None

    def build(self, vector_dim):
        """ returns a word2vec model """
        logging.info("Building keras model")
        word_input = Input(shape=(1,), name="word_input")
        word = Embedding(input_dim=self.data.vocab_size, output_dim=vector_dim, input_length=1, name="word_embedding")(
            word_input)
        # word = Reshape((vector_dim, 1))(word)

        context_input = Input(shape=(1,), name="context_input")
        context = Embedding(input_dim=self.data.vocab_size, output_dim=vector_dim, input_length=1,
                            name="context_embedding")(
            context_input)

        merged = dot([word, context], axes=2, normalize=False, name="dot")
        merged = Flatten()(merged)
        output = Dense(1, activation='sigmoid', name="output")(merged)

        optimizer = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
        model = Model(inputs=[word_input, context_input], outputs=output)
        model.compile(loss="binary_crossentropy", optimizer=optimizer)
        self.model = model

    def train(self, window_size, negative_samples, epochs, verbose):
        """ Trains the word2vec model """
        logging.info("Training model")

        # in order to balance out more negative samples than positive
        negative_weight = 1.0 / negative_samples
        logging.info("Setting negative weight to: %.4f", negative_weight)
        class_weight = {1: 1.0, 0: negative_weight}

        # rough estimation of the train set size
        steps_per_epoch = self.data.size * 2 * window_size * negative_samples
        sampling_table = sequence.make_sampling_table(self.data.vocab_size)
        skip_gram_iterator = self.data.skip_gram_iterator(window_size, negative_samples, shuffle=True,
                                                          sampling_table=sampling_table)
        self.model.fit_generator(skip_gram_iterator,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 verbose=verbose,
                                 class_weight=class_weight,
                                 max_queue_size=10000)

    def write_embeddings(self, path):
        """ Saves the embeddings of a file """
        logging.info("Saving embeddings to %s", path)
        with open(path, 'wb') as fout:
            for i in xrange(len(self.data.index2Word)):
                word = self.data.index2Word[i]
                weights = self.model.get_layer("word_embedding").get_weights()[0][i]
                fout.write('%d\t' % word)
                fout.write(np.array2string(weights, precision=4, separator=',', prefix="", max_line_width=10000))
                fout.write("\n")
            fout.flush()

    def plot(self, path):
        """ Plots the model to a file"""
        logging.info("Saving model image to: %s", path)
        plot_model(self.model, show_shapes=True, to_file=path)
