import argparse
import json
import logging
import os

import numpy as np


def predict():
    parser = argparse.ArgumentParser(description='A word2vec implementation in Keras')
    parser.add_argument('--embeddings', type=str, help='Path to output embeddings folder', required=True)
    parser.add_argument('--word', type=str, help='Word to predict similar words to', required=True)
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

    word2index = dict((v, int(k)) for k, v in index2word.iteritems())

    word = args.word
    if word not in word2index:
        logging.info("Could not find word %s in index", word)
        return

    word_embedding = embeddings[word2index[word]]
    candidates = {}
    for i in xrange(len(embeddings)):
        candidate_embedding = embeddings[i]
        score = np.dot(word_embedding, candidate_embedding)
        candidate = index2word["%d" % i]
        candidates[candidate] = score

    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1])
    for i in xrange(10):
        candidate, score = sorted_candidates[i]
        logging.info("%d: %s %.4f", i + 1, candidate, score)


if __name__ == "__main__":
    predict()
