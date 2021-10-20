import os
import argparse

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Iterable

from scripts import SPACY_PROCESSOR
from main import run, OUTPUT_FOLDER, export


def get_kmeans_cluster_labels(data: list, num_of_clusters: int):
    """
    Method for getting silhouette score and cluster labels
    :param data: data for clustering
    :param num_of_clusters: number of clusters
    :return: tuple: number of clusters, score, cluster labels
    """
    kmeans_words = KMeans(n_clusters=num_of_clusters)
    kmeans_words.fit(data)
    new_cluster_labels = kmeans_words.labels_

    return num_of_clusters, silhouette_score(data, new_cluster_labels), new_cluster_labels


def cluster_words_kmeans(data: Iterable, num_of_clusters: int = 10) -> pd.DataFrame:
    """
    Function for clustering words into groups
    :param data: iterable list or sequence of words
    :param num_of_clusters: number of clusters
    :return: DataFrame of clusters
    """
    data = SPACY_PROCESSOR(' '.join(data))
    data = pd.DataFrame([(x.text, x.vector) for x in data], columns=['word', 'vector'])
    vector_data = np.array([*data['vector']])

    res = [get_kmeans_cluster_labels(vector_data, x) for x in range(2, num_of_clusters + 1)]

    # assigning cluster for each word
    data['cluster'] = max(res, key=lambda x: x[1])[2]
    clusters = [df['word'].tolist() for label, df in data.groupby('cluster')]
    return pd.DataFrame([(cluster,) for cluster in clusters], columns=['cluster'])


@run
def main():
    def read_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument('--word_threshold', '-w', help='Word threshold for clusters', type=int, required=False,
                            default=100)
        parser.add_argument('--input_path', '-i', help='Full path for input file', type=str, required=True)
        parser.add_argument('--output_path', '-o', help=f'Name for output file, it will be created in {OUTPUT_FOLDER}',
                            type=str, required=True)
        return parser.parse_args()

    args = read_args()
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
    if args.input_path.find('.json') < 0:
        raise Exception('input file must have json extension')

    data = pd.read_json(args.input_path)
    res = cluster_words_kmeans(data['word'])
    export(res, os.path.join(OUTPUT_FOLDER, args.output_path))


if __name__ == '__main__':
    main()
