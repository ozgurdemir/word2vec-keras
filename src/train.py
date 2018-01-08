import argparse
import logging
import numpy as np

import data
from word2vec import Word2Vec


def train():
    parser = argparse.ArgumentParser(description='A word2vec implementation in Keras')
    parser.add_argument('--train', type=str, help='Path to train set', required=True)
    parser.add_argument('--image', type=str, help='Path to model image')
    parser.add_argument('--embeddings', type=str, help='Path to output embeddings')
    parser.add_argument('--threshold', type=int, help='Words occurring less than this are removed', default=5)
    parser.add_argument('--windowSize', type=int, help='Skip gram window size', default=4)
    parser.add_argument('--negatives', type=int, help='Number of negative samples per input word', default=5)
    parser.add_argument('--batchSize', type=int, help='Training batch size', default=512)
    parser.add_argument('--epochs', type=int, help='Training epochs', default=1)
    parser.add_argument('--verbose', type=int, help='Verbosity', default=1)
    parser.add_argument('--vectorDim', type=int, help='Size of the embeddings', default=50)
    parser.add_argument('--learnRate', type=float, help='Initial learn rate', default=0.1)
    args = parser.parse_args()

    log_format = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)

    logging.info(args)

    sequence = data.read(file_name=args.train)
    word_occurrence = data.word_dict(sequence)
    word_occurrence = data.prune_occurrence(word_occurrence, args.threshold)
    word2index, index2word = data.word_index(word_occurrence)
    sequence = data.re_index(sequence, word2index)
    sequence = np.asarray(sequence, dtype=np.int)

    vocab_size = len(word_occurrence)

    word_2_vec = Word2Vec()
    word_2_vec.build(args.vectorDim, vocab_size, args.learnRate)

    if args.image:
        word_2_vec.plot(args.image)

    word_2_vec.train(sequence, args.windowSize, args.negatives, args.batchSize, args.epochs, args.verbose)

    if args.embeddings:
        data.write_embeddings(args.embeddings, index2word,
                              word_2_vec.model.get_layer("word_embedding").get_weights()[0])


if __name__ == "__main__":
    train()
