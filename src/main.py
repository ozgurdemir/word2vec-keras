import argparse
import logging

from data import Data
from word2vec import Word2Vec


def main():
    parser = argparse.ArgumentParser(description='A word2vec implementation in keras')
    parser.add_argument('--train', type=str, help='Path to train set', required=True)
    parser.add_argument('--image', type=str, help='Path to model image')
    parser.add_argument('--embeddings', type=str, help='Path to output embeddings file')
    args = parser.parse_args()

    log_format = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)

    data = Data(file_name=args.train)
    data.word_dict()
    data.word_index()

    word_2_vec = Word2Vec(data)
    word_2_vec.build(vector_dim=150)

    if args.image:
        word_2_vec.plot(args.image)

    word_2_vec.train(window_size=5, negative_samples=1, epochs=1, verbose=1)

    if args.embeddings:
        word_2_vec.write_embeddings(args.embeddings)


if __name__ == "__main__":
    main()
