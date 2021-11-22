"""
Module for preprocessing data
Initial filtering, data modification, etc.
"""
import string
import pandas as pd
import numpy as np

from typing import List

from sklearn.base import BaseEstimator, TransformerMixin

from ..utils import run


class Preprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, pipeline: List[str], column: str):
        self.pipeline = pipeline
        self.column = column

    @run
    def drop_empty(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function for dropping empty rows by column value
        """
        df[self.column].replace('', np.nan, inplace=True)
        df.dropna(subset=[self.column], inplace=True)
        return df

    @run
    def lowercase(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function for lower casing column data
        """
        df[self.column] = df[self.column].str.lower()
        return df

    @run
    def drop_duplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function for dropping duplicate values by column
        """
        return df.drop_duplicates(subset=[self.column])

    @run
    def remove_punctuation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Function for removing punctuation symbols
        """
        df[self.column] = df[self.column].str.translate(str.maketrans('', '', string.punctuation))
        return df

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method for transforming data
        """
        for pipe in self.pipeline:
            df = getattr(self, pipe)(df)
        return df[self.column]
