"""
Module with core functions for processing data
"""

import numpy
import pandas as pd
import numpy as np

from typing import List
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from spacy.tokens import Token

from ..utils import run


@run
def count_words(tokens: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    """
    Function that creates DataFrame of token count
    tokens DataFrame example:
            word
        0   abc
        1   abc
        2   example
            ...
    result DataFrame example:
            word    count
        0   abc     2
        1   example 1
            ...     ...
    """
    count_series = pd.Series([token for token in tokens['text']]).value_counts()
    return pd.DataFrame({'word': count_series.index, 'count': count_series.values})


@run
def cluster_words_kmeans(df: pd.DataFrame, num_of_clusters: int = 10) -> pd.DataFrame:
    """
    Function for clustering words into groups using KMeans method
    example input DataFrame:
            text    vector
        0   abc     [...]
        1   example [...]
            ...     ...
    example output DataFrame:
            cluster
        0   ['abc', ...]
        1   ['example', ...]
            ...
    """

    def get_kmeans_cluster_labels(data: numpy.ndarray, num_of_clusters: int):
        """
        Method for getting silhouette score and cluster labels
        """
        kmean_tokens = KMeans(n_clusters=num_of_clusters)
        kmean_tokens.fit(data)
        cluster_labels = kmean_tokens.labels_

        return num_of_clusters, silhouette_score(data, cluster_labels), cluster_labels

    vector_data = np.array([*df['vector']])

    res = [get_kmeans_cluster_labels(vector_data, x) for x in range(2, num_of_clusters + 1)]

    # assigning cluster for each word, by determining best cluster number
    df['cluster'] = max(res, key=lambda x: x[1])[2]
    clusters = [df['text'].tolist() for label, df in df.groupby('cluster')]
    return pd.DataFrame([(cluster,) for cluster in clusters], columns=['cluster'])


@run
def create_core(percent_threshold: float, tokens: List[Token], *args, **kwargs) -> list:
    """
    Function for creating core of tokens by comparing them and filtering by threshold
    """
    res = list()

    # comparing word pairs
    percent_threshold = percent_threshold / 100
    for token_1 in tokens:
        for token_2 in tokens:
            if token_1.similarity(token_2) < percent_threshold:
                break
        else:
            res.append(token_1.text)

    return res
