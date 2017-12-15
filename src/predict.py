import argparse
import json
import logging
import os

import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial.distance import cosine


def predict():
    parser = argparse.ArgumentParser(description='A word2vec implementation in Keras')
    parser.add_argument('--embeddings', type=str, help='Path to output embeddings', required=True)
    args = parser.parse_args()

    log_format = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename='example.log', format=log_format, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info("Loading embeddings from %s", args.embeddings)
    embeddings = pd.read_csv(args.embeddings, index_col=0)

    query = embeddings.loc['as']

    while True:
        word = input("""Please enter a word (enter "EXIT" to quit): """)
        if word == "EXIT":
            return
        elif word not in embeddings.index:
            print("Could not find word '%s' in index" % word)
        else:
            query = embeddings.loc[word]
            similarities = embeddings.apply(lambda x: 1.0 - cosine(query, x), axis=1)
            similarities.sort_values(inplace=True, ascending=False)
            print(similarities[0:10])


if __name__ == "__main__":
    predict()
