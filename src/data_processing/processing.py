"""
Module with core functions for processing data
"""
import os
import numpy
import pandas as pd
import numpy as np

from typing import List

from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from . import BasePipe


class TokenProcessor(BasePipe):
    """
    Class for processing spacy tokens
    """

    def __init__(self, pipeline: List[str], percent_threshold: float, num_of_clusters: int = 10,
                 training_dataset_path: str = None):
        super().__init__(pipeline)
        self.num_of_clusters = num_of_clusters
        self.training_dataset_path = training_dataset_path
        self.percent_threshold = percent_threshold

    def cluster_words_kmeans(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function for clustering words into groups using KMeans method
        example input DataFrame:
                text    ... vector
            0   abc     ... [...]
            1   example ... [...]
                ...     ... ...
        example output DataFrame:
                text    ... vector  cluster
            0   abc     ... [...]   1
            1   example ... [...]   2
                ...     ... ...     ...
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

        res = [get_kmeans_cluster_labels(vector_data, x) for x in range(2, self.num_of_clusters + 1)]

        # assigning cluster for each word, by determining best cluster number
        df['cluster'] = max(res, key=lambda x: x[1])[2]
        return df

    def cluster_words_svm(self, df: pd.DataFrame):
        """
        Function for clustering words into groups using SVM method
        example input DataFrame:
                text    ... vector
            0   abc     ... [...]
            1   example ... [...]
                ...     ... ...
        example output DataFrame:
                text    ... vector  cluster
            0   abc     ... [...]   1
            1   example ... [...]   2
                ...     ... ...     ...
        """
        if not os.path.exists(self.training_dataset_path):
            raise FileNotFoundError(f'{self.training_dataset_path} - no such file.')
        training_dataset = pd.read_json(self.training_dataset_path)
        clf = svm.SVC()
        clf.fit(np.array([*training_dataset['vector']]), training_dataset['cluster'])
        clusters = clf.predict(np.array([*df['vector']]))
        df['cluster'] = clusters
        return df

    def create_core(self, df: pd.DataFrame) -> list:
        """
        Function for creating core of tokens by comparing them and filtering by threshold
        """
        res = list()

        def token_similarity(vector: np.array, other_vector: np.array, vector_norm: float,
                             other_vector_norm: float) -> float:
            """
            Function for comparing tokens using vectors
            """
            return np.dot(vector, other_vector) / (vector_norm * other_vector_norm)

        # comparing word pairs
        percent_threshold = self.percent_threshold / 100
        for _, token_1 in df.iterrows():
            for _, token_2 in df.iterrows():
                if token_similarity(token_1['vector'], token_2['vector'], token_1['vector_norm'],
                                    token_2['vector_norm']) < percent_threshold:
                    break
            else:
                res.append(token_1['text'])

        return res
