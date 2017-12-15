import argparse
import json
import logging
import os
from scipy import spatial

import numpy as np


def predict():
    parser = argparse.ArgumentParser(description='A word2vec implementation in Keras')
    parser.add_argument('--embeddings', type=str, help='Path to output embeddings folder', required=True)
    args = parser.parse_args()

    log_format = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename='example.log', format=log_format, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    embeddings_path = os.path.join(args.embeddings, "embeddings.csv")
    logging.info("Loading embeddings from %s", embeddings_path)
    embeddings = np.loadtxt(embeddings_path)

    index2word_path = os.path.join(args.embeddings, "index2word.txt")
    logging.info("Loading index2word from %s", index2word_path)
    with open(index2word_path, 'r') as f:
        index2word = dict(json.load(f))

    word2index = dict((v, int(k)) for k, v in index2word.items())

    while True:
        word = input("""Please enter a word (enter "EXIT" to quit): """)
        if word == "EXIT":
            return
        elif word not in word2index:
            logging.info("Could not find word %s in index", word)
        else:
            word_embedding = embeddings[word2index[word]]
            candidates = {}
            for i in range(len(embeddings)):
                candidate_embedding = embeddings[i]
                score = 1 - spatial.distance.cosine(word_embedding, candidate_embedding)
                candidate = index2word["%d" % i]
                candidates[candidate] = score

            sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
            for i in range(10):
                candidate, score = sorted_candidates[i]
                logging.info("%d: %s %.4f", i + 1, candidate, score)


if __name__ == "__main__":
    predict()
