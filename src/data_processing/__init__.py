import pandas as pd

from typing import List

from sklearn.base import BaseEstimator, TransformerMixin


class BasePipe(BaseEstimator, TransformerMixin):
    """
    Base pipe implementation
    """

    def __init__(self, pipeline: List[str]):
        self.pipeline = pipeline

    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method for running pipe functions
        """
        for pipe in self.pipeline:
            df = getattr(self, pipe)(df)
        return df

    def fit(self, X, y=None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method for transforming data
        """
        for pipe in self.pipeline:
            df = getattr(self, pipe)(df)
        return df
