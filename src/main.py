import logging
import argparse

from word2vec import Word2Vec
from data import Data


def main():
    parser = argparse.ArgumentParser(description='A word2vec implementation in keras')
    parser.add_argument('--train', type=str, help='Path to train set', required=True)
    parser.add_argument('--image', type=str, help='Path to model image')
    parser.add_argument('--embeddings', type=str, help='Path to output embeddings file')
    args = parser.parse_args()

    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename='../example.log', format=FORMAT, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    data = Data(file_name=args.train)
    data.word_dict()
    data.buildIndex()

    word_2_vec = Word2Vec(data)
    word_2_vec.build(vector_dim=150)

    if args.image:
        word_2_vec.plot(args.image)

    word_2_vec.train(args.train, negative_samples=5, epochs=1, window_size=10, max_batch=2)

    if args.embeddings:
        word_2_vec.write_embeddings(args.embeddings)


if __name__ == "__main__":
    main()
