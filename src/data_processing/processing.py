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
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils import run


class SpacyTokenProcessor(BaseEstimator, TransformerMixin):

    def __init__(self, pipeline: List[str], percent_threshold: float, num_of_clusters: int = 10,
                 training_dataset_path: str = None):
        self.pipeline = pipeline
        self.num_of_clusters = num_of_clusters
        self.training_dataset_path = training_dataset_path
        self.percent_threshold = percent_threshold

    @run
    def count_words(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function that creates DataFrame of token count
        tokens DataFrame example:
                text    ...
            0   abc     ...
            1   abc     ...
            2   example ...
                ...     ...
        result DataFrame example:
                word    ... count
            0   abc     ... 2
            1   example ... 1
                ...     ... ...
        """
        count_series = df['text'].value_counts()
        df = df.drop_duplicates(subset=['text'])
        df['count'] = df.apply(lambda token: count_series.at[token['text']], axis=1)
        return df

    @run
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

    @run
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

    @run
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

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method for transforming data
        """
        for pipe in self.pipeline:
            df = getattr(self, pipe)(df)
        return df
