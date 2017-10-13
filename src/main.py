import logging

from word2vec import Word2Vec
from src.data import Data


def main():
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename='../example.log', format=FORMAT, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    train_set = "../data/trainset.txt"
    data = Data(file_name=train_set)
    data.word_dict()
    data.buildIndex()

    word_2_vec = Word2Vec(data)
    word_2_vec.build(vector_dim=150)
    word_2_vec.plot("../data/model.png")
    word_2_vec.train(train_set, negative_samples=5, epochs=1, window_size=10, max_batch=2)
    word_2_vec.write_embeddings("../data/embeddings.txt")


if __name__ == "__main__":
    main()
